import pygame
import math
import random
import sys

# --- Configuración inicial ---
WIDTH, HEIGHT = 1000, 800
CENTER = (WIDTH // 2, HEIGHT // 2)
NUM_STARS = 400
G_I = 500  # Constante de gravedad informacional
ENTROPY_FACTOR = 0.0015
TIME_STEP = 0.1

# Inicializa Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulador G_I: Evolución Galáctica")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)

# --- Estrellas ---
class Star:
    def __init__(self):
        self.radius = random.uniform(30, 280)
        self.angle = random.uniform(0, 2 * math.pi)
        self.info_density = 1 / (self.radius + 1)
        self.speed = math.sqrt(G_I * self.info_density)
        self.entropy = random.uniform(-1, 1)
        self.size = random.uniform(1.0, 2.5)
        self.color = pygame.Color(0)
        self.color.hsva = (random.randint(0, 360), 100, 100, 100)

    def update(self):
        self.angle += self.speed * TIME_STEP
        self.entropy += ENTROPY_FACTOR * (random.random() - 0.5)
        self.radius += self.entropy * 0.1
        self.radius = max(1, self.radius)
        self.info_density = 1 / (self.radius + 1)
        self.speed = math.sqrt(G_I * self.info_density)

    def draw(self, surface):
        x = CENTER[0] + math.cos(self.angle) * self.radius
        y = CENTER[1] + math.sin(self.angle) * self.radius
        pygame.draw.circle(surface, self.color, (int(x), int(y)), int(self.size))

# --- Lista de estrellas ---
stars = [Star() for _ in range(NUM_STARS)]

# --- Bucle principal ---
running = True
while running:
    screen.fill((0, 0, 0))

    for star in stars:
        star.update()
        star.draw(screen)

    # Mostrar parámetros en pantalla
    text = f"G_I: {G_I:.2f} | Entropía: {ENTROPY_FACTOR:.5f} | Tiempo: {TIME_STEP:.2f}"
    label = font.render(text, True, (255, 255, 255))
    screen.blit(label, (20, 20))

    pygame.display.flip()
    clock.tick(60)

    # Gestión de eventos
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Controles de teclado
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                G_I += 10
            elif event.key == pygame.K_a:
                G_I = max(0, G_I - 10)
            elif event.key == pygame.K_w:
                ENTROPY_FACTOR += 0.0005
            elif event.key == pygame.K_s:
                ENTROPY_FACTOR = max(0.0001, ENTROPY_FACTOR - 0.0005)
            elif event.key == pygame.K_e:
                TIME_STEP += 0.01
            elif event.key == pygame.K_d:
                TIME_STEP = max(0.01, TIME_STEP - 0.01)

pygame.quit()
sys.exit()
