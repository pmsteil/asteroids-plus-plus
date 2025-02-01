import pygame
import sys
import random
import math
import os
import numpy as np
import array

# Game constants
WIDTH, HEIGHT = 800, 600
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Approximate collision radius for the ship (tweak as needed)
SHIP_COLLISION_RADIUS = 15

# Sound effects
class SoundEffects:
    def __init__(self):
        self.sounds = {}
        self._create_sounds()
        self.thrust_playing = False
        self.beat_tempo = 1.0
        self.last_beat_time = 0
        
    def _create_sounds(self):
        # Create synthesized sounds using pygame
        sample_rate = 44100
        max_amplitude = 32767  # Max value for 16-bit audio
        
        def clamp(value):
            """Clamp value to valid 16-bit range"""
            return max(min(value, 32767), -32768)
        
        def create_buffer(samples):
            """Convert numpy array to proper audio buffer"""
            buffer = array.array('h')  # signed short integer array
            buffer.extend(samples)
            return buffer
        
        # Fire sound (softer, lower pitched beep)
        duration = 0.05  # shorter duration
        num_samples = int(duration * sample_rate)
        samples = array.array('h', [0] * (num_samples * 2))  # Stereo, so *2
        for i in range(num_samples):
            t = float(i) / sample_rate
            value = clamp(int(max_amplitude * 0.15 * math.sin(2.0 * math.pi * 220.0 * t)))
            samples[i * 2] = value  # Left channel
            samples[i * 2 + 1] = value  # Right channel
        self.sounds['fire'] = pygame.mixer.Sound(buffer=samples)
        self.sounds['fire'].set_volume(0.3)
        
        # Thrust sound (low rumble)
        duration = 1.0
        num_samples = int(duration * sample_rate)
        samples = array.array('h', [0] * (num_samples * 2))
        for i in range(num_samples):
            t = float(i) / sample_rate
            value = clamp(int(max_amplitude * 0.25 * (
                math.sin(2.0 * math.pi * 100.0 * t) +
                0.5 * math.sin(2.0 * math.pi * 80.0 * t)
            )))
            samples[i * 2] = value
            samples[i * 2 + 1] = value
        self.sounds['thrust'] = pygame.mixer.Sound(buffer=samples)
        self.sounds['thrust'].set_volume(0.4)
        
        # Explosion sounds
        def create_explosion(base_freq, duration, volume=0.7):
            num_samples = int(duration * sample_rate)
            samples = array.array('h', [0] * (num_samples * 2))
            for i in range(num_samples):
                t = float(i) / sample_rate
                decay = math.exp(-3.0 * t)
                value = clamp(int(max_amplitude * volume * decay * (
                    0.5 * math.sin(2.0 * math.pi * base_freq * t) +
                    0.3 * math.sin(2.0 * math.pi * (base_freq * 1.5) * t) +
                    0.2 * math.sin(2.0 * math.pi * (base_freq * 0.5) * t) +
                    0.1 * random.uniform(-1, 1)  # Noise
                )))
                samples[i * 2] = value
                samples[i * 2 + 1] = value
            sound = pygame.mixer.Sound(buffer=samples)
            sound.set_volume(0.7)
            return sound
            
        self.sounds['big_explosion'] = create_explosion(80, 0.6)
        self.sounds['medium_explosion'] = create_explosion(120, 0.5)
        self.sounds['small_explosion'] = create_explosion(160, 0.4)
        
        # Ship explosion (more dramatic)
        duration = 0.8
        num_samples = int(duration * sample_rate)
        samples = array.array('h', [0] * (num_samples * 2))
        for i in range(num_samples):
            t = float(i) / sample_rate
            decay = math.exp(-2.0 * t)
            value = clamp(int(max_amplitude * decay * (
                0.4 * math.sin(2.0 * math.pi * 60.0 * t) +
                0.3 * math.sin(2.0 * math.pi * 90.0 * t) +
                0.2 * math.sin(2.0 * math.pi * 30.0 * t) +
                0.3 * random.uniform(-1, 1)
            )))
            samples[i * 2] = value
            samples[i * 2 + 1] = value
        self.sounds['ship_explosion'] = pygame.mixer.Sound(buffer=samples)
        self.sounds['ship_explosion'].set_volume(0.8)
        
        # Beat sound (low thump)
        duration = 0.1
        num_samples = int(duration * sample_rate)
        samples = array.array('h', [0] * (num_samples * 2))
        for i in range(num_samples):
            t = float(i) / sample_rate
            decay = math.exp(-10.0 * t)
            value = clamp(int(max_amplitude * 0.3 * decay * math.sin(2.0 * math.pi * 50.0 * t)))
            samples[i * 2] = value
            samples[i * 2 + 1] = value
        self.sounds['beat'] = pygame.mixer.Sound(buffer=samples)
        self.sounds['beat'].set_volume(0.4)
        
        # Extra life sound (high pitched jingle)
        duration = 0.5
        num_samples = int(duration * sample_rate)
        samples = array.array('h', [0] * (num_samples * 2))
        for i in range(num_samples):
            t = float(i) / sample_rate
            freq = 440.0 * (1 + t)
            value = clamp(int(max_amplitude * 0.3 * math.sin(2.0 * math.pi * freq * t)))
            samples[i * 2] = value
            samples[i * 2 + 1] = value
        self.sounds['extra_life'] = pygame.mixer.Sound(buffer=samples)
        self.sounds['extra_life'].set_volume(0.5)
    
    def play(self, sound_name):
        if sound_name in self.sounds:
            self.sounds[sound_name].play()
    
    def start_thrust(self):
        if not self.thrust_playing:
            self.sounds['thrust'].play(-1)  # Loop indefinitely
            self.thrust_playing = True
    
    def stop_thrust(self):
        if self.thrust_playing:
            self.sounds['thrust'].stop()
            self.thrust_playing = False
    
    def update_beat(self, asteroid_count):
        # Increase tempo based on remaining asteroids
        current_time = pygame.time.get_ticks()
        self.beat_tempo = 1.0 + (20 - min(20, asteroid_count)) * 0.1
        if current_time - self.last_beat_time > 1000 / self.beat_tempo:
            self.play('beat')
            self.last_beat_time = current_time

class Ship:
    def __init__(self, pos, angle=0):
        self.pos = pygame.Vector2(pos)
        self.angle = angle  # in degrees
        self.velocity = pygame.Vector2(0, 0)
        self.acceleration = 0.2
        self.friction = 0.99
        # Define the ship as a triangle (points relative to the center)
        self.points = [pygame.Vector2(0, -15), pygame.Vector2(10, 10), pygame.Vector2(-10, 10)]
        self.invulnerable = False
        self.invulnerable_timer = 0

    def update(self):
        self.pos += self.velocity
        self.velocity *= self.friction
        self.wrap()
        if self.invulnerable:
            self.invulnerable_timer -= 1
            if self.invulnerable_timer <= 0:
                self.invulnerable = False

    def make_invulnerable(self, frames=120):  # 2 seconds at 60 FPS
        self.invulnerable = True
        self.invulnerable_timer = frames

    def draw(self, surface):
        # If invulnerable, blink the ship
        if self.invulnerable and (self.invulnerable_timer // 4) % 2:
            return
        # Rotate and translate the ship's points
        rad = math.radians(self.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        transformed = []
        for point in self.points:
            x = point.x * cos_a - point.y * sin_a
            y = point.x * sin_a + point.y * cos_a
            transformed.append((x + self.pos.x, y + self.pos.y))
        pygame.draw.polygon(surface, WHITE, transformed, 1)

    def accelerate(self):
        # Accelerate in the direction the ship is pointing
        rad = math.radians(self.angle)
        force = pygame.Vector2(math.sin(rad), -math.cos(rad)) * self.acceleration
        self.velocity += force

    def rotate(self, direction):
        # direction: -1 for left, +1 for right
        self.angle += 5 * direction

    def wrap(self):
        # Wrap around screen edges
        if self.pos.x > WIDTH:
            self.pos.x = 0
        elif self.pos.x < 0:
            self.pos.x = WIDTH
        if self.pos.y > HEIGHT:
            self.pos.y = 0
        elif self.pos.y < 0:
            self.pos.y = HEIGHT

class Asteroid:
    def __init__(self, pos, size):
        self.pos = pygame.Vector2(pos)
        self.size = size  # radius of the asteroid
        angle = random.uniform(0, 360)
        speed = random.uniform(1, 3)
        self.velocity = pygame.Vector2(math.cos(math.radians(angle)), math.sin(math.radians(angle))) * speed
        self.points = self.generate_points()

    def generate_points(self):
        # Create a rough polygon to represent the asteroid
        points = []
        num_points = random.randint(8, 12)
        for i in range(num_points):
            angle = i * (360 / num_points) + random.uniform(-10, 10)
            rad = math.radians(angle)
            distance = self.size + random.uniform(-self.size * 0.4, self.size * 0.4)
            x = math.cos(rad) * distance
            y = math.sin(rad) * distance
            points.append((x, y))
        return points

    def update(self):
        self.pos += self.velocity
        self.wrap()

    def wrap(self):
        if self.pos.x > WIDTH:
            self.pos.x = 0
        elif self.pos.x < 0:
            self.pos.x = WIDTH
        if self.pos.y > HEIGHT:
            self.pos.y = 0
        elif self.pos.y < 0:
            self.pos.y = HEIGHT

    def draw(self, surface):
        # Translate the asteroid's polygon points to its position
        transformed = [(self.pos.x + x, self.pos.y + y) for (x, y) in self.points]
        pygame.draw.polygon(surface, WHITE, transformed, 1)

class Bullet:
    def __init__(self, pos, angle):
        self.pos = pygame.Vector2(pos)
        rad = math.radians(angle)
        self.velocity = pygame.Vector2(math.sin(rad), -math.cos(rad)) * 10
        self.lifetime = 60  # frames bullet will be alive

    def update(self):
        self.pos += self.velocity
        self.lifetime -= 1
        self.wrap()

    def wrap(self):
        if self.pos.x > WIDTH:
            self.pos.x = 0
        elif self.pos.x < 0:
            self.pos.x = WIDTH
        if self.pos.y > HEIGHT:
            self.pos.y = 0
        elif self.pos.y < 0:
            self.pos.y = HEIGHT

    def draw(self, surface):
        pygame.draw.circle(surface, RED, (int(self.pos.x), int(self.pos.y)), 2)

class Particle:
    def __init__(self, pos, velocity, lifetime=30):
        self.pos = pygame.Vector2(pos)
        self.velocity = pygame.Vector2(velocity)
        self.lifetime = lifetime
        self.original_lifetime = lifetime

    def update(self):
        self.pos += self.velocity
        self.lifetime -= 1
        # Slow down the particle
        self.velocity *= 0.95
        return self.lifetime > 0

    def draw(self, surface):
        # Fade out as lifetime decreases
        alpha = int((self.lifetime / self.original_lifetime) * 255)
        color = (255, min(255, 128 + alpha), 0, alpha)
        pygame.draw.circle(surface, color, (int(self.pos.x), int(self.pos.y)), 1)

def draw_ship_icon(surface, pos):
    # Draw a small ship icon (triangle) at the given position
    points = [pygame.Vector2(0, -8), pygame.Vector2(6, 8), pygame.Vector2(-6, 8)]
    transformed = [(p.x + pos[0], p.y + pos[1]) for p in points]
    pygame.draw.polygon(surface, WHITE, transformed, 0)

def create_explosion_particles(pos, count=20):
    particles = []
    for _ in range(count):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 5)
        velocity = (math.cos(angle) * speed, math.sin(angle) * speed)
        particles.append(Particle(pos, velocity))
    return particles

def draw_ui(surface, score, lives, game_over, level=1, show_level_text=False):
    font = pygame.font.SysFont('Arial', 24)
    # Draw score at top left
    score_text = font.render(f"Score: {score}", True, WHITE)
    surface.blit(score_text, (10, 10))

    # Draw level at top center
    level_text = font.render(f"Level {level}", True, WHITE)
    level_rect = level_text.get_rect(midtop=(WIDTH/2, 10))
    surface.blit(level_text, level_rect)

    # Draw lives icons at upper right
    for i in range(lives):
        icon_x = WIDTH - (i + 1) * 30
        icon_y = 10
        draw_ship_icon(surface, (icon_x, icon_y + 10))

    # If showing level announcement
    if show_level_text:
        level_font = pygame.font.SysFont('Arial', 48)
        announce_text = level_font.render(f'Level {level}', True, WHITE)
        text_rect = announce_text.get_rect(center=(WIDTH / 2, HEIGHT / 2))
        surface.blit(announce_text, text_rect)

    # If game over, display GAME OVER message and restart instruction
    if game_over:
        game_font = pygame.font.SysFont('Arial', 48)
        game_over_text = game_font.render('GAME OVER', True, WHITE)
        text_rect = game_over_text.get_rect(center=(WIDTH / 2, HEIGHT / 2))
        surface.blit(game_over_text, text_rect)
        
        restart_font = pygame.font.SysFont('Arial', 24)
        restart_text = restart_font.render('Press R to Restart', True, WHITE)
        restart_rect = restart_text.get_rect(center=(WIDTH / 2, HEIGHT / 2 + 50))
        surface.blit(restart_text, restart_rect)

def main():
    pygame.init()
    pygame.mixer.init(44100, -16, 2, 1024)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Asteroids")
    clock = pygame.time.Clock()

    def reset_ship():
        ship = Ship((WIDTH / 2, HEIGHT / 2))
        ship.make_invulnerable()
        return ship

    def init_game():
        nonlocal ship, lives, score, in_game, game_over, particles, asteroids, bullets, level, level_start_time, show_level_text
        ship = reset_ship()
        lives = 3
        score = 0
        level = 1
        in_game = True
        game_over = False
        particles = []
        level_start_time = pygame.time.get_ticks()
        show_level_text = True
        # Start with base number of asteroids
        start_new_level(level)

    def start_new_level(level_num):
        nonlocal asteroids, level_start_time, show_level_text
        asteroids = []
        # Base number of asteroids is 5, increases by 10% each level (rounded up)
        num_asteroids = math.ceil(5 * (1 + (level_num - 1) * 0.1))
        # Asteroids get slightly faster each level
        base_speed = 1 + (level_num - 1) * 0.1
        for _ in range(num_asteroids):
            # Spawn asteroids away from the ship
            while True:
                pos = pygame.Vector2(random.randrange(WIDTH), random.randrange(HEIGHT))
                if pos.distance_to(ship.pos) > 100:  # Minimum safe distance
                    break
            size = random.randint(20, 40)
            new_asteroid = Asteroid(pos, size)
            # Increase asteroid speed based on level
            new_asteroid.velocity *= base_speed
            asteroids.append(new_asteroid)
        level_start_time = pygame.time.get_ticks()
        show_level_text = True

    # Initialize game variables
    ship = None
    lives = 0
    score = 0
    level = 1
    in_game = True
    game_over = False
    particles = []
    asteroids = []
    bullets = []
    level_start_time = 0
    show_level_text = False
    init_game()  # Initialize all game variables

    # Initialize sound effects
    sound_effects = SoundEffects()

    running = True
    while running:
        clock.tick(FPS)
        current_time = pygame.time.get_ticks()

        # Handle level text display
        if show_level_text and current_time - level_start_time > 2000:  # Show for 2 seconds
            show_level_text = False

        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and game_over:
                    init_game()
                elif event.key == pygame.K_SPACE and in_game:
                    bullets.append(Bullet(ship.pos, ship.angle))
                    sound_effects.play('fire')

        if in_game:
            # Check if level is complete
            if len(asteroids) == 0:
                level += 1
                sound_effects.play('extra_life')  # Use as level-up sound
                start_new_level(level)

            # Update particles
            particles = [p for p in particles if p.update()]

            # Update background beat
            sound_effects.update_beat(len(asteroids))

            # Handle continuous key presses for rotation and acceleration
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                ship.rotate(-1)
            if keys[pygame.K_RIGHT]:
                ship.rotate(1)
            if keys[pygame.K_UP]:
                ship.accelerate()
                sound_effects.start_thrust()
            else:
                sound_effects.stop_thrust()

            # Update game objects
            ship.update()
            for asteroid in asteroids:
                asteroid.update()
            for bullet in bullets[:]:
                bullet.update()
                if bullet.lifetime <= 0:
                    bullets.remove(bullet)

            # Check for bullet-asteroid collisions
            for bullet in bullets[:]:
                for asteroid in asteroids[:]:
                    if bullet.pos.distance_to(asteroid.pos) < asteroid.size:
                        try:
                            bullets.remove(bullet)
                        except ValueError:
                            pass
                        try:
                            asteroids.remove(asteroid)
                        except ValueError:
                            pass
                        score += 100
                        
                        # Play appropriate explosion sound based on asteroid size
                        if asteroid.size > 30:
                            sound_effects.play('big_explosion')
                        elif asteroid.size > 15:
                            sound_effects.play('medium_explosion')
                        else:
                            sound_effects.play('small_explosion')
                            
                        if asteroid.size > 15:
                            for _ in range(2):
                                new_size = asteroid.size // 2
                                new_asteroid = Asteroid(asteroid.pos, new_size)
                                # Inherit parent asteroid's speed scaling
                                new_asteroid.velocity *= asteroid.velocity.length() / 2
                                asteroids.append(new_asteroid)
                        break

            # Check for ship collisions
            if not ship.invulnerable:
                for asteroid in asteroids:
                    if ship.pos.distance_to(asteroid.pos) < (asteroid.size + SHIP_COLLISION_RADIUS):
                        print("Collision detected! Ship hit an asteroid.")
                        lives -= 1
                        
                        # Create explosion particles and play sound immediately
                        particles.extend(create_explosion_particles(ship.pos))
                        particles.extend(create_explosion_particles(ship.pos, 30))  # Add more particles
                        life_icon_pos = (WIDTH - lives * 30 - 30, 20)
                        particles.extend(create_explosion_particles(life_icon_pos, 10))
                        sound_effects.play('ship_explosion')
                        
                        if lives > 0:
                            # Create the new ship but keep it invisible briefly
                            ship = reset_ship()
                            ship.invulnerable = True
                            ship.invulnerable_timer = 90  # 1.5 seconds at 60 FPS
                        else:
                            in_game = False
                            game_over = True
                        break

        # Render the scene
        screen.fill(BLACK)
        if in_game:
            ship.draw(screen)
            for asteroid in asteroids:
                asteroid.draw(screen)
            for bullet in bullets:
                bullet.draw(screen)
        # Draw particles
        for particle in particles:
            particle.draw(screen)
        draw_ui(screen, score, lives, game_over, level, show_level_text)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
