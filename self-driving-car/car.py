"""Car class for the self-driving car simulation."""

import math

import pygame

from sensor import Sensor
from utils import polys_intersect


class Car:
    """Represents a car in the simulation, driven by keys or an AI brain."""

    def __init__(self, x, y, width=40, height=70, controller="KEYS"):
        """Initialize a car at position (x, y).

        Args:
            x: Initial x position
            y: Initial y position
            width: Car width
            height: Car height
            controller: "KEYS" for keyboard control, "AI" for brain control
        """
        self.x = x
        self.y = y
        self.start_y = y
        self.width = width
        self.height = height

        self.speed = 0
        self.acceleration = 0.2
        self.max_speed = 3
        self.friction = 0.05
        self.angle = 0

        self.damaged = False
        self.finished = False
        self.fitness = 0

        # Extra per-car stats, tracked for logging/analysis rather than
        # anything the sim itself needs.
        self.frames_alive = 0
        self.max_speed_reached = 0.0
        self.speed_sum = 0.0
        self.turn_left_frames = 0
        self.turn_right_frames = 0

        self.controller = controller
        self.controls = {"forward": False, "left": False, "right": False, "reverse": False}

        self.brain = None
        self.sensor = Sensor(self) if controller == "AI" else None

    @property
    def avg_speed(self):
        """Average absolute speed over the car's lifetime so far."""
        return self.speed_sum / self.frames_alive if self.frames_alive else 0.0

    def update(self, road_borders, obstacles):
        """Advance the car by one simulation step.

        Args:
            road_borders: List of [start, end] segments for the road edges
            obstacles: List of Obstacle instances to collide with
        """
        if self.damaged or self.finished:
            return

        if self.sensor:
            self.sensor.update(road_borders, obstacles)

        self.control()
        self.move()

        self.frames_alive += 1
        self.max_speed_reached = max(self.max_speed_reached, abs(self.speed))
        self.speed_sum += abs(self.speed)
        if self.controls["left"]:
            self.turn_left_frames += 1
        if self.controls["right"]:
            self.turn_right_frames += 1

        if self.assess_damage(road_borders, obstacles):
            self.damaged = True

    def get_corners(self):
        """Compute the car's four corners as (x, y) tuples, accounting
        for its current rotation.
        """
        radius = math.hypot(self.width, self.height) / 2
        alpha = math.atan2(self.width, self.height)
        angle = math.radians(self.angle)

        corners = []
        for offset in (alpha, -alpha, math.pi + alpha, math.pi - alpha):
            corners.append((
                self.x - math.sin(angle + offset) * radius,
                self.y - math.cos(angle + offset) * radius,
            ))
        return corners

    def assess_damage(self, road_borders, obstacles):
        """Check whether the car's polygon overlaps a border or obstacle.

        Args:
            road_borders: List of [start, end] segments for the road edges
            obstacles: List of Obstacle instances

        Returns:
            bool: True if the car has crashed
        """
        polygon = self.get_corners()

        for border in road_borders:
            if polys_intersect(polygon, border):
                return True

        for obstacle in obstacles:
            if polys_intersect(polygon, obstacle.polygon):
                return True

        return False

    def move(self):
        """Update speed, angle, and position based on current controls."""
        if self.controls["forward"]:
            self.speed += self.acceleration
        if self.controls["reverse"]:
            self.speed -= self.acceleration

        self.speed = max(-self.max_speed / 2, min(self.max_speed, self.speed))

        if self.speed > 0:
            self.speed = max(0, self.speed - self.friction)
        elif self.speed < 0:
            self.speed = min(0, self.speed + self.friction)

        if self.controls["left"]:
            self.angle += 3
        if self.controls["right"]:
            self.angle -= 3

        radians = math.radians(self.angle)
        self.x -= math.sin(radians) * self.speed
        self.y -= math.cos(radians) * self.speed

        self.fitness = self.start_y - self.y

    def draw(self, screen, offset_y=0, color=None):
        """Draw the car on the given pygame surface.

        Args:
            screen: pygame.Surface to draw on
            offset_y: Vertical camera offset to apply
            color: Optional RGB override; defaults by damaged state
        """
        if color is None:
            color = (200, 60, 60) if self.damaged else (0, 200, 255)

        points = [(x, y + offset_y) for x, y in self.get_corners()]
        pygame.draw.polygon(screen, color, points)

    def control(self):
        """Set self.controls from either the keyboard or the AI brain.

        Controls:
            UP: Increase speed
            DOWN: Decrease speed
            LEFT: Rotate counter-clockwise
            RIGHT: Rotate clockwise
        """
        if self.controller == "AI" and self.brain is not None:
            outputs = self.brain.feed_forward(self.sensor.readings)
            self.controls["forward"] = outputs[0] > 0.5
            self.controls["left"] = outputs[1] > 0.5
            self.controls["right"] = outputs[2] > 0.5
            self.controls["reverse"] = outputs[3] > 0.5
            return

        keys = pygame.key.get_pressed()
        self.controls["forward"] = bool(keys[pygame.K_UP])
        self.controls["reverse"] = bool(keys[pygame.K_DOWN])
        self.controls["left"] = bool(keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT])
        self.controls["right"] = bool(keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT])
