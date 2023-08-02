import pygame
import random
from Network import *

pygame.init()
pygame.display.set_caption("Project: Neural Networks: Image Processing")


class Graphical_Interface_Element:
    def __init__(self):
        self.x = None
        self.y = None
        self.w = None
        self.h = None
        pass

    def contains(self, mouse_pos):
        return self.x <= mouse_pos[0] <= (self.x + self.w) and self.y <= mouse_pos[1] <= (self.y + self.h)

    def draw(self, screen):
        pass


class Text(Graphical_Interface_Element):
    def __init__(self, text, font, color):
        super().__init__()
        self.text_string = text
        self.font = font
        self.color = color
        self.text = font.render(self.text_string, True, color)
        self.rect = None
        self.rect_kwargs = None

    def set_rect(self, **kwargs):
        self.rect = self.text.get_rect(**kwargs)
        self.rect_kwargs = kwargs
        self.x = self.rect.left
        self.y = self.rect.top
        self.w = self.rect.right - self.rect.left
        self.h = self.rect.bottom - self.rect.top

    def set_text_string(self, string):
        self.text_string = string
        self.text = self.font.render(self.text_string, True, self.color)
        self.set_rect(**self.rect_kwargs)

    def set_color(self, color):
        self.color = color
        self.text = self.font.render(self.text_string, True, self.color)

    def draw(self, screen):
        if self.text is not None and self.rect is not None:
            screen.blit(self.text, self.rect)


class Button(Text):
    def __init__(self, text, font, color):
        super().__init__(text, font, color)

    def draw(self, screen):
        button_pos = (self.x - 2, self.y - 2, self.w + 4, self.h + 4)
        pygame.draw.rect(screen, WHITE, button_pos)
        pygame.draw.rect(screen, WHITE if dark_mode else BLACK, button_pos, 1)
        screen.blit(self.text, self.rect)


class Canvas(Graphical_Interface_Element):
    def __init__(self, rows, columns, box_dim, canvas_pos=(None, None)):
        super().__init__()
        self.rows = rows
        self.columns = columns
        self.pixels = [[0]*self.columns for _ in range(self.rows)]
        self.box_w = box_dim[0]
        self.box_h = box_dim[1]
        self.w = self.columns * self.box_w
        self.h = self.rows * self.box_h
        self.x = int((SCREEN_WIDTH - self.w) / 2) if canvas_pos[0] is None else canvas_pos[0]
        self.y = int((SCREEN_HEIGHT - self.h) / 2) if canvas_pos[1] is None else canvas_pos[1]

    def draw(self, screen):
        for row in range(0, self.rows):
            for col in range(0, self.columns):
                color = (int(abs(self.pixels[row][col] if dark_mode else self.pixels[row][col] - 255)),) * 3
                if color == BLACK and dark_mode or color == WHITE and not dark_mode:
                    continue

                pixel_x = self.x + self.box_w * col
                pixel_y = self.y + self.box_h * row
                pygame.draw.rect(screen, color, (pixel_x, pixel_y, self.box_w, self.box_h))

        for col in range(self.columns + 1):
            pygame.draw.line(screen, L_GRAY if dark_mode else D_GRAY, (self.x + (col * self.box_w), self.y), (self.x + (col * self.box_w), self.y + self.h))

        for row in range(self.rows + 1):
            pygame.draw.line(screen, L_GRAY if dark_mode else D_GRAY, (self.x, self.y + (row * self.box_h)), (self.x + self.w, self.y + (row * self.box_h)))

    def paint(self, mouse_pos):
        mouse_x = mouse_pos[0] - self.x
        mouse_y = mouse_pos[1] - self.y
        canvas_col = max(min(int(mouse_x/self.box_w), self.columns - 1), 0)
        canvas_row = max(min(int(mouse_y/self.box_h), self.rows - 1), 0)
        self.pixels[canvas_row][canvas_col] = min(255, self.pixels[canvas_row][canvas_col] + 100)

        for row in range(canvas_row - 1, canvas_row + 1):
            for col in range(canvas_col - 1, canvas_col + 1):
                if row == canvas_row and col == canvas_col:
                    continue
                if 0 <= row < self.rows and 0 <= col < self.columns:
                    row_off = abs(canvas_row - row)
                    col_off = abs(canvas_col - col)
                    self.pixels[row][col] = max(min(255, self.pixels[row][col] + random.randint(25, 100) - (0 * (row_off + col_off))), 0)

    def reset_paint(self):
        self.pixels = [[0]*self.columns for _ in range(self.rows)]

    def set_paint(self, pixels):
        self.pixels = pixels


# Network
network = Network(None, "MNISTuner.txt")

# Data
test_data = load("mnist_test.csv", None, True)
sorted_test_data = [[], [], [], [], [], [], [], [], [], []]

for sample in test_data:
    sorted_test_data[int(sample[0])].append(sample[1:785])

# Pixels
ROWS = 28
COLS = 28

# Screen
SCREEN_WIDTH, SCREEN_HEIGHT = 650, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Colors
BLACK = (0, 0, 0)
L_GRAY = (227, 227, 227)
D_GRAY = (181, 181, 181)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

# Mode
dark_mode = True

# Font
font = pygame.font.Font("Retro Gaming.ttf", 30)
medium_font = pygame.font.Font("Retro Gaming.ttf", 17)
small_font = pygame.font.Font("Retro Gaming.ttf", 14)

# GUI Elements: Title
title = Text("THE IMAGE PROCESSOR", font, BLUE)
title.set_rect(center=(SCREEN_WIDTH/2, 50))

# Some Pizaz
title_shadow = Text("THE IMAGE PROCESSOR", font, RED)
title_shadow.set_rect(topleft=(title.rect.left - 1, title.rect.top + 1))

# GUI Elements: Button: Reset
reset = Button("Reset Image", medium_font, BLACK)
reset.set_rect(topright=(SCREEN_WIDTH/2 - 10, title.rect.bottom + 10))

# GUI Elements: Button: Test
test = Button("Test Image", medium_font, BLACK)
test.set_rect(topleft=(SCREEN_WIDTH/2 + 10, title.rect.bottom + 10))

# GUI Elements: Canvas
canvas = Canvas(ROWS, COLS, (15, 15), (None, test.rect.bottom + 25))

# GUI Elements: Text: Output
output = Text("Draw an Image", medium_font, WHITE)
output.set_rect(centerx=SCREEN_WIDTH/2, top=canvas.y+canvas.h+25)

# Cursor
pygame.mouse.set_cursor(*pygame.cursors.arrow)

run = True
drag = False

while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            if test.contains(pos):
                p_input = np.divide(np.array(canvas.pixels).flatten(), 255)
                out = network.get_output(p_input)
                confidence = out.max()
                prediction = out.argmax()
                output.set_text_string(f"Bob is {(float(confidence) * 100):.2f}% sure that this is a {prediction}.")
            elif reset.contains(pos):
                canvas.reset_paint()
                output.set_text_string("Draw an Image")
            elif canvas.contains(pos):
                canvas.paint(pos)
            drag = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drag = False
        if event.type == pygame.MOUSEMOTION:
            pos = pygame.mouse.get_pos()
            if canvas.contains(pos):
                pygame.mouse.set_cursor(*pygame.cursors.ball)
                if drag:
                    canvas.paint(pos)
            elif test.contains(pos) or reset.contains(pos):
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
            else:
                pygame.mouse.set_cursor(*pygame.cursors.arrow)
        if event.type == pygame.KEYDOWN:
            sample_input = None
            keys = pygame.key.get_pressed()

            if keys[pygame.K_0]:
                sample_input = sorted_test_data[0]
            elif keys[pygame.K_1]:
                sample_input = sorted_test_data[1]
            elif keys[pygame.K_2]:
                sample_input = sorted_test_data[2]
            elif keys[pygame.K_3]:
                sample_input = sorted_test_data[3]
            elif keys[pygame.K_4]:
                sample_input = sorted_test_data[4]
            elif keys[pygame.K_5]:
                sample_input = sorted_test_data[5]
            elif keys[pygame.K_6]:
                sample_input = sorted_test_data[6]
            elif keys[pygame.K_7]:
                sample_input = sorted_test_data[7]
            elif keys[pygame.K_8]:
                sample_input = sorted_test_data[8]
            elif keys[pygame.K_9]:
                sample_input = sorted_test_data[9]

            if sample_input is not None:
                sample_paint = sample_input[random.randint(0, len(sample_input) - 1)]
                canvas.set_paint(np.array(sample_paint).reshape(28, 28))

            if keys[pygame.K_t]:
                dark_mode = not dark_mode
                output.set_color(WHITE if dark_mode else BLACK)

    screen.fill(BLACK if dark_mode else WHITE)

    # Drawing Title
    title_shadow.draw(screen)
    title.draw(screen)

    # Drawing Buttons
    reset.draw(screen)
    test.draw(screen)

    # Drawing Canvas
    canvas.draw(screen)

    # Drawing Output
    output.draw(screen)

    pygame.display.update()