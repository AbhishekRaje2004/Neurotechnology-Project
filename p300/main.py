import pygame
import random
import time
import csv

pygame.init()

screen_info = pygame.display.Info()
window_width, window_height = screen_info.current_w, screen_info.current_h  # Unpack the screen dimensions

board_size = 6 
square_size = 100  
spacing = 5 

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
START_COLOR = (50, 50, 50)  # Dark gray for the start area

screen = pygame.display.set_mode((window_width, window_height), pygame.FULLSCREEN)
pygame.display.set_caption('Chessboard Flipper for P300')

def draw_chessboard():
    x_offset = (window_width - (board_size * square_size + (board_size - 1) * spacing)) // 2
    y_offset = (window_height - (board_size * square_size + (board_size - 1) * spacing)) // 2
    
    for row in range(board_size):
        for col in range(board_size):
            color = WHITE if (row + col) % 2 == 0 else BLACK
            pygame.draw.rect(screen, color, (x_offset + col * (square_size + spacing), 
                                             y_offset + row * (square_size + spacing), 
                                             square_size, square_size))

def flip_chessboard(trigger_times, trial, flip):
    flip_time = time.time()
    trigger_times.append(flip_time)
    
    x_offset = (window_width - (board_size * square_size + (board_size - 1) * spacing)) // 2
    y_offset = (window_height - (board_size * square_size + (board_size - 1) * spacing)) // 2
    
    for row in range(board_size):
        for col in range(board_size):

            color = BLACK if (row + col) % 2 == 0 else WHITE
            pygame.draw.rect(screen, color, (x_offset + col * (square_size + spacing), 
                                             y_offset + row * (square_size + spacing), 
                                             square_size, square_size))
    
    pygame.display.flip()  
    time.sleep(0.5) 
    
    draw_chessboard()
    pygame.display.flip()
    time.sleep(0.8) 

def show_starting_slide():

    screen.fill(BLACK)


    start_rect = pygame.Rect((window_width // 4), (window_height // 2 - 50), window_width // 2, 100)
    pygame.draw.rect(screen, START_COLOR, start_rect)


    font = pygame.font.SysFont('Arial', 36)
    text = font.render("Press SPACE to start", True, WHITE)
    text_rect = text.get_rect(center=start_rect.center)
    screen.blit(text, text_rect)

    pygame.display.flip()


    waiting_for_key = True
    while waiting_for_key:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting_for_key = False
                screen.fill(BLACK)  
                pygame.display.flip()  

def run_experiment():
    running = True
    show_starting_slide() 
    draw_chessboard()  
    pygame.display.flip()


    trigger_times = []


    for trial in range(1):  
        print(f"Starting Trial {trial + 1}")
        for flip in range(10): 
            interval = random.uniform(1.0, 3.0)
            time.sleep(interval)  
            flip_chessboard(trigger_times, trial, flip)


            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    return 
    with open('trigger_times.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Trial", "Flip", "Time"])
        for trial in range(5):
            for flip, flip_time in enumerate(trigger_times[trial*20:(trial+1)*20]):
                writer.writerow([trial+1, flip+1, flip_time])
    
    pygame.quit()  

run_experiment()
