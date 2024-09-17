import pygame
import random
import numpy as np
from training import predict

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 420, 420
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Movable Grayscale Drawing Board with Prediction")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)

# Drawing variables
drawing = False
moving = False
last_pos = None
brush_size = 15
fontObj = pygame.font.Font(None, 32)
input_size = 280  # Size of the drawing area
output_size = 28
margin = (width - input_size) // 2  # Margin for centering

# Create a surface for drawing
drawing_surface = pygame.Surface((input_size, input_size))
drawing_surface.fill(BLACK)

# Offset for moving the image
offset_x, offset_y = 0, 0


def draw_line(surface, start, end, radius, color):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = max(abs(dx), abs(dy))

    for i in range(distance):
        x = int(start[0] + float(i) / distance * dx)
        y = int(start[1] + float(i) / distance * dy)
        pygame.draw.circle(surface, color, (x, y), radius)


# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                drawing = True
                last_pos = (event.pos[0] - margin - offset_x, event.pos[1] - margin - offset_y)
            elif event.button == 3:  # Right mouse button
                moving = True
                last_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                drawing = False
            elif event.button == 3:  # Right mouse button
                moving = False
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                current_pos = (event.pos[0] - margin - offset_x, event.pos[1] - margin - offset_y)
                if last_pos:
                    if 0 <= current_pos[0] < input_size and 0 <= current_pos[1] < input_size:
                        draw_line(drawing_surface, last_pos, current_pos, brush_size // 2, WHITE)
                        # Add grayscale effect
                        for x in range(max(0, min(last_pos[0], current_pos[0])),
                                       min(input_size, max(last_pos[0], current_pos[0]) + 1)):
                            for y in range(max(0, min(last_pos[1], current_pos[1])),
                                           min(input_size, max(last_pos[1], current_pos[1]) + 1)):
                                if pygame.math.Vector2(x - last_pos[0], y - last_pos[1]).length() <= brush_size / 2:
                                    gray = random.randint(200, 255)
                                    drawing_surface.set_at((x, y), (gray, gray, gray))
                last_pos = current_pos
            elif moving:
                current_pos = event.pos
                dx = current_pos[0] - last_pos[0]
                dy = current_pos[1] - last_pos[1]
                offset_x += dx
                offset_y += dy
                last_pos = current_pos
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:  # Press 'C' to clear the screen
                drawing_surface.fill(BLACK)
                offset_x, offset_y = 0, 0

    # Clear the main screen and draw the centered drawing area
    screen.fill(GRAY)
    screen.blit(drawing_surface, (margin + offset_x, margin + offset_y))

    # Draw a border around the drawing area
    pygame.draw.rect(screen, WHITE, (margin - 1, margin - 1, input_size + 2, input_size + 2), 1)

    # Predict the drawn digit
    pic = pygame.surfarray.array3d(drawing_surface)
    pic = np.mean(pic, axis=2)  # Convert to grayscale
    bin_size = input_size // output_size
    small = pic.reshape((output_size, bin_size, output_size, bin_size)).max(3).max(1)
    a = predict(small.flatten() / 255.0)
    pr = fontObj.render(str(a), True, WHITE)
    pygame.draw.rect(screen, BLACK, (margin, margin - 40, 32, 32))
    screen.blit(pr, (margin, margin - 40))

    pygame.display.flip()

pygame.quit()