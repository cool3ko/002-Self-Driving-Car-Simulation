"""Static obstacles placed on the road."""

import pygame


class Obstacle:
    """A rectangular obstacle the car must steer around."""

    def __init__(self, x, y, width=30, height=30, color=(160, 60, 40)):
        """Initialize an obstacle centered at (x, y).

        Args:
            x: Center x position
            y: Center y position
            width: Obstacle width
            height: Obstacle height
            color: RGB fill color
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color

    @property
    def polygon(self):
        """Return the obstacle's four corners as (x, y) tuples."""
        half_w = self.width / 2
        half_h = self.height / 2
        return [
            (self.x - half_w, self.y - half_h),
            (self.x + half_w, self.y - half_h),
            (self.x + half_w, self.y + half_h),
            (self.x - half_w, self.y + half_h),
        ]

    def draw(self, screen, offset_y=0):
        """Draw the obstacle on the given surface.

        Args:
            screen: pygame.Surface to draw on
            offset_y: Vertical camera offset to apply
        """
        points = [(x, y + offset_y) for x, y in self.polygon]
        pygame.draw.polygon(screen, self.color, points)
