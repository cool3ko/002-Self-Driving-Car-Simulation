"""Sensor class for detecting road edges and obstacles."""

import math

import pygame

from utils import line_intersect


class Sensor:
    """A fan of rays cast from the car, used as the brain's eyes."""

    def __init__(self, car, ray_count=7, ray_length=220, ray_spread=math.pi / 2):
        """Initialize a sensor for the given car.

        Args:
            car: Car object to attach sensor to
            ray_count: Number of rays in the fan
            ray_length: Length of each ray
            ray_spread: Total angle (radians) the fan spans
        """
        self.car = car
        self.ray_count = ray_count
        self.ray_length = ray_length
        self.ray_spread = ray_spread
        self.rays = []
        self.readings = [0.0] * ray_count

    def update(self, road_borders, obstacles):
        """Recompute ray positions and readings for the car's current pose.

        Args:
            road_borders: List of [start, end] segments for the road edges
            obstacles: List of Obstacle instances
        """
        self.rays = [self._cast_ray(i) for i in range(self.ray_count)]
        self.readings = [
            self._get_reading(ray, road_borders, obstacles) for ray in self.rays
        ]

    def _cast_ray(self, index):
        fraction = 0.5 if self.ray_count == 1 else index / (self.ray_count - 1)
        angle_offset = self.ray_spread / 2 - self.ray_spread * fraction
        angle = math.radians(self.car.angle) + angle_offset

        start = (self.car.x, self.car.y)
        end = (
            self.car.x - math.sin(angle) * self.ray_length,
            self.car.y - math.cos(angle) * self.ray_length,
        )
        return start, end

    def _get_reading(self, ray, road_borders, obstacles):
        start, end = ray
        touches = []

        for border in road_borders:
            hit = line_intersect(start, end, border[0], border[1])
            if hit:
                touches.append(hit)

        for obstacle in obstacles:
            polygon = obstacle.polygon
            for i in range(len(polygon)):
                hit = line_intersect(
                    start, end, polygon[i], polygon[(i + 1) % len(polygon)]
                )
                if hit:
                    touches.append(hit)

        if not touches:
            return 0.0

        closest = min(touches, key=lambda hit: hit[2])
        return 1 - closest[2]

    def draw(self, screen, offset_y=0):
        """Draw the sensor rays on the given pygame surface.

        Args:
            screen: pygame.Surface to draw on
            offset_y: Vertical camera offset to apply
        """
        for i, (start, end) in enumerate(self.rays):
            reading = self.readings[i] if i < len(self.readings) else 0.0
            color = (255, 60, 60) if reading > 0 else (255, 220, 0)
            pygame.draw.line(
                screen,
                color,
                (start[0], start[1] + offset_y),
                (end[0], end[1] + offset_y),
                2,
            )
