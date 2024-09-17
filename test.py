import numpy as np
import pygame
import random
import math
a=np.array([1,2,3,4])
b=np.array([2,2,2,2])
a=a.reshape(4,1)
b=b.reshape(4,1)
print(a.dot(b.T))
print(np.outer(a,b))
print(np.outer(b,a))
print(np.add(b,np.full(4,1)))
print(np.full((4,1),1))


# Initialize Pygame
pygame.init()

# Set up the display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Grayscale Drawing Board with Warring Colors")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Drawing variables
drawing = False
last_pos = None
brush_size = 5

# Warring color particles
particles = []


class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.speed = random.uniform(1, 3)
        self.direction = random.uniform(0, 2 * 3.14159)

    def move(self):
        self.x += self.speed * math.cos(self.direction)
        self.y += self.speed * math.sin(self.direction)

        # Bounce off edges
        if self.x < 0 or self.x > width:
            self.direction = 3.14159 - self.direction
        if self.y < 0 or self.y > height:
            self.direction = -self.direction

    def draw(self):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), 2)


# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            last_pos = pygame.mouse.get_pos()
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.MOUSEMOTION and drawing:
            current_pos = pygame.mouse.get_pos()
            if last_pos:
                pygame.draw.line(screen, WHITE, last_pos, current_pos, brush_size)
                # Add grayscale effect
                for x in range(min(last_pos[0], current_pos[0]), max(last_pos[0], current_pos[0]) + 1):
                    for y in range(min(last_pos[1], current_pos[1]), max(last_pos[1], current_pos[1]) + 1):
                        if pygame.math.Vector2(x - last_pos[0], y - last_pos[1]).length() <= brush_size / 2:
                            gray = random.randint(0, 255)
                            screen.set_at((x, y), (gray, gray, gray))
            last_pos = current_pos

    # Move and draw particles
    for particle in particles:
        particle.move()
        particle.draw()

    # Randomly add new particles
    if random.random() < 0.1:
        color = RED if random.random() < 0.5 else BLUE
        particles.append(Particle(random.randint(0, width), random.randint(0, height), color))

    # Remove particles that are too old
    particles = [p for p in particles if 0 <= p.x <= width and 0 <= p.y <= height]

    pygame.display.flip()

pygame.quit()
