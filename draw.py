import pygame
import numpy as np
from training import predict

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 140, 140
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Smooth Grayscale Drawing Board with Prediction")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Drawing variables
drawing = False
last_pos = None
brush_size = 10
fontObj = pygame.font.Font(None, 32)
input_size = 140
output_size = 28


def draw_line(surface, start, end, radius, color):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = max(abs(dx), abs(dy))

    for i in range(distance):
        x = int(start[0] + float(i) / distance * dx)
        y = int(start[1] + float(i) / distance * dy)
        pygame.draw.circle(surface, color, (x, y), radius)
def center_image(image):
    # Find the bounding box of the non-zero regions
    non_zero = np.argwhere(image > 0)
    if len(non_zero) == 0:
        return image  # Return original image if it's all zeros

    top, left = non_zero.min(axis=0)
    bottom, right = non_zero.max(axis=0)

    # Calculate the center of the bounding box
    center_y, center_x = (top + bottom) // 2, (left + right) // 2

    # Calculate the translation needed to center the image
    dy = image.shape[0] // 2 - center_y
    dx = image.shape[1] // 2 - center_x

    # Create a new centered image
    centered = np.zeros_like(image)
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            new_y, new_x = y + dy, x + dx
            if 0 <= new_y < image.shape[0] and 0 <= new_x < image.shape[1]:
                centered[new_y, new_x] = image[y, x]

    return centered

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            last_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.MOUSEMOTION and drawing:
            current_pos = event.pos
            if last_pos:
                draw_line(screen, last_pos, current_pos, brush_size // 2, WHITE)
                # Add grayscale effect
                for x in range(min(last_pos[0], current_pos[0]), max(last_pos[0], current_pos[0]) + 1):
                    for y in range(min(last_pos[1], current_pos[1]), max(last_pos[1], current_pos[1]) + 1):
                        if pygame.math.Vector2(x - last_pos[0], y - last_pos[1]).length() <= brush_size / 2:
                            gray = np.random.randint(180, 255)  # Increased minimum to 200 for better visibility
                            screen.set_at((x, y), (gray, gray, gray))
            last_pos = current_pos
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:  # Press 'C' to clear the screen
                screen.fill(BLACK)

    # Predict the drawn digit
    pic = pygame.surfarray.array3d(screen)
    for i in range(0, 20):
        for j in range(0, 15):
            pic[i][j] = 0
    pic = np.mean(pic, axis=2)  # Convert to grayscale
    bin_size = input_size // output_size
    small =pic.reshape((output_size, bin_size, output_size, bin_size)).max(3).max(1).T
    a = predict(small.flatten() / 255.0)
    pr = fontObj.render(str(a), True, WHITE)
    pygame.draw.rect(screen, BLACK, (0, 0, 32, 32))
    screen.blit(pr, (0, 0))

    pygame.display.flip()

pygame.quit()