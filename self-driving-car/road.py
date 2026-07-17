"""Road class for the simulation environment."""

import pygame

INFINITY = 1_000_000


class Road:
    """Represents a multi-lane road with lane markings and hard edges."""

    def __init__(self, x, width, lane_count=3):
        """Initialize a road.

        Args:
            x: Center x position of the road
            width: Width of the road
            lane_count: Number of lanes the road is divided into
        """
        self.x = x
        self.width = width
        self.lane_count = lane_count
        self.left = self.x - self.width / 2
        self.right = self.x + self.width / 2

        self.color = (100, 100, 100)
        self.line_color = (255, 255, 255)
        self.line_width = 5
        self.dash_length = 20
        self.dash_spacing = 40

        # The road extends far beyond the screen in both directions so
        # collisions work no matter how far the car has driven.
        top_left = (self.left, -INFINITY)
        bottom_left = (self.left, INFINITY)
        top_right = (self.right, -INFINITY)
        bottom_right = (self.right, INFINITY)
        self.borders = [
            [top_left, bottom_left],
            [top_right, bottom_right],
        ]

    def get_lane_center(self, lane_index):
        """Return the x coordinate of the center of a lane.

        Args:
            lane_index: Zero-based lane index, clamped to valid lanes

        Returns:
            float: x coordinate of the lane's center
        """
        lane_index = max(0, min(self.lane_count - 1, lane_index))
        lane_width = self.width / self.lane_count
        return self.left + lane_width / 2 + lane_index * lane_width

    def draw(self, surface, offset_y=0):
        """Draw the road on the given pygame surface.

        Args:
            surface: pygame.Surface to draw on
            offset_y: Vertical camera offset, used to scroll the
                dashed lane markings as the car advances
        """
        window_height = surface.get_height()

        # Draw road surface
        pygame.draw.rect(
            surface,
            self.color,
            (self.left, 0, self.width, window_height)
        )

        # Draw left/right road edges
        pygame.draw.line(
            surface, self.line_color, (self.left, 0), (self.left, window_height), self.line_width
        )
        pygame.draw.line(
            surface, self.line_color, (self.right, 0), (self.right, window_height), self.line_width
        )

        # Draw dashed lane dividers, scrolling with the camera
        lane_width = self.width / self.lane_count
        phase = offset_y % self.dash_spacing
        for lane_index in range(1, self.lane_count):
            divider_x = self.left + lane_index * lane_width
            y = phase - self.dash_spacing
            while y < window_height:
                pygame.draw.line(
                    surface,
                    self.line_color,
                    (divider_x, y),
                    (divider_x, y + self.dash_length),
                    3
                )
                y += self.dash_spacing
