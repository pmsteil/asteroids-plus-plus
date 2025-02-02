import pygame
import sys
import random
import math
import os
import numpy as np
import array
import json
from pathlib import Path

# Game constants
DEFAULT_WIDTH, DEFAULT_HEIGHT = 800, 600
ASPECT_RATIO = DEFAULT_WIDTH / DEFAULT_HEIGHT
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# File paths
HIGH_SCORES_FILE = Path(__file__).parent / "high_scores.json"

class Game:
    def __init__(self):
        pygame.init()
        pygame.mixer.init(44100, -16, 2, 1024)

        # Get the display info
        info = pygame.display.Info()
        self.max_width = info.current_w
        self.max_height = info.current_h

        # Set initial window size to 80% of screen size while maintaining aspect ratio
        target_width = int(self.max_width * 0.8)
        target_height = int(target_width / ASPECT_RATIO)
        if target_height > self.max_height * 0.8:
            target_height = int(self.max_height * 0.8)
            target_width = int(target_height * ASPECT_RATIO)

        self.width = target_width
        self.height = target_height

        # Create resizable window
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("Asteroids")

        # Scale factor for game objects (relative to default size)
        self.scale_x = self.width / DEFAULT_WIDTH
        self.scale_y = self.height / DEFAULT_HEIGHT

        # Approximate collision radius for the ship (scaled)
        self.ship_collision_radius = 15 * self.scale_x

        self.clock = pygame.time.Clock()
        self.high_scores = HighScores()
        self.sound_effects = SoundEffects()
        self.reset_game_state()

    def reset_game_state(self):
        """Initialize or reset all game state variables"""
        self.ship = self.reset_ship()
        self.lives = 3
        self.score = 0
        self.level = 1
        self.in_game = True
        self.game_over = False
        self.entering_name = False
        self.current_name = ""
        self.particles = []
        self.asteroids = []
        self.bullets = []
        self.level_start_time = pygame.time.get_ticks()
        self.show_level_text = True
        self.start_new_level(self.level)

    def handle_resize(self, new_width, new_height):
        """Handle window resize event"""
        # Maintain aspect ratio
        target_height = int(new_width / ASPECT_RATIO)
        if target_height > new_height:
            new_width = int(new_height * ASPECT_RATIO)
        else:
            new_height = target_height

        self.width = new_width
        self.height = new_height
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)

        # Update scale factors
        self.scale_x = self.width / DEFAULT_WIDTH
        self.scale_y = self.height / DEFAULT_HEIGHT
        self.ship_collision_radius = 15 * self.scale_x

        # Scale existing game objects
        if self.ship:
            self.ship.pos.x = (self.ship.pos.x / self.scale_x) * self.scale_x
            self.ship.pos.y = (self.ship.pos.y / self.scale_y) * self.scale_y

        for asteroid in self.asteroids:
            asteroid.pos.x = (asteroid.pos.x / self.scale_x) * self.scale_x
            asteroid.pos.y = (asteroid.pos.y / self.scale_y) * self.scale_y
            asteroid.size = asteroid.original_size * self.scale_x

        for bullet in self.bullets:
            bullet.pos.x = (bullet.pos.x / self.scale_x) * self.scale_x
            bullet.pos.y = (bullet.pos.y / self.scale_y) * self.scale_y

    def reset_ship(self):
        """Create a new ship in the center of the screen"""
        ship = Ship((self.width / 2, self.height / 2), scale=(self.scale_x, self.scale_y))
        ship.make_invulnerable()
        return ship

    def start_new_level(self, level_num):
        """Initialize a new level"""
        self.asteroids = []
        # Base number of asteroids is 5, increases by 10% each level (rounded up)
        num_asteroids = math.ceil(5 * (1 + (level_num - 1) * 0.1))
        # Asteroids get slightly faster each level
        base_speed = 1 + (level_num - 1) * 0.1

        for _ in range(num_asteroids):
            # Spawn asteroids away from the ship
            while True:
                pos = pygame.Vector2(random.randrange(self.width), random.randrange(self.height))
                if pos.distance_to(self.ship.pos) > 100 * self.scale_x:  # Minimum safe distance
                    break
            size = random.randint(20, 40)
            new_asteroid = Asteroid(pos, size, scale=(self.scale_x, self.scale_y))
            new_asteroid.velocity *= base_speed
            self.asteroids.append(new_asteroid)

        self.level_start_time = pygame.time.get_ticks()
        self.show_level_text = True

    def run(self):
        """Main game loop"""
        running = True
        while running:
            self.clock.tick(FPS)
            current_time = pygame.time.get_ticks()

            # Handle level text display
            if self.show_level_text and current_time - self.level_start_time > 2000:
                self.show_level_text = False

            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.VIDEORESIZE:
                    self.handle_resize(event.w, event.h)

                elif event.type == pygame.KEYDOWN:
                    if self.game_over:
                        if self.entering_name:
                            if event.key == pygame.K_RETURN and self.current_name.strip():
                                self.high_scores.add_score(self.current_name, self.score)
                                self.entering_name = False
                            elif event.key == pygame.K_BACKSPACE:
                                self.current_name = self.current_name[:-1]
                            elif len(self.current_name) < 10 and event.unicode.isalnum():
                                self.current_name += event.unicode
                        else:
                            if event.key == pygame.K_r:
                                self.reset_game_state()
                            elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                                running = False
                    elif event.key == pygame.K_SPACE and self.in_game:
                        self.bullets.append(Bullet(self.ship.get_nose_position(), self.ship.angle, scale=(self.scale_x, self.scale_y)))
                        self.sound_effects.play('fire')

            if self.in_game:
                # Check if level is complete
                if len(self.asteroids) == 0:
                    self.level += 1
                    self.sound_effects.play('extra_life')
                    self.start_new_level(self.level)

                # Update particles
                self.particles = [p for p in self.particles if p.update()]

                # Update background beat
                self.sound_effects.update_beat(len(self.asteroids))

                # Handle continuous key presses
                keys = pygame.key.get_pressed()
                thrust = keys[pygame.K_UP]
                rotate = 0
                if keys[pygame.K_LEFT]:
                    rotate = -1
                elif keys[pygame.K_RIGHT]:
                    rotate = 1

                # Update ship and get thruster particles
                if self.ship:
                    thruster_particles = self.ship.update(thrust, rotate, self.width, self.height)
                    if thruster_particles:
                        self.particles.extend(thruster_particles)

                # Update sound effects
                if thrust:
                    self.sound_effects.start_thrust()
                else:
                    self.sound_effects.stop_thrust()

                # Update game objects
                for asteroid in self.asteroids:
                    asteroid.update(self.width, self.height)  # Pass screen dimensions
                for bullet in self.bullets[:]:
                    bullet.update(self.width, self.height)  # Pass screen dimensions
                    if bullet.lifetime <= 0:
                        self.bullets.remove(bullet)

                # Handle collisions
                self.handle_collisions()

            # Render the scene
            self.screen.fill(BLACK)
            if self.in_game:
                if self.ship:
                    self.ship.draw(self.screen)
                for asteroid in self.asteroids:
                    asteroid.draw(self.screen)
                for bullet in self.bullets:
                    bullet.draw(self.screen)

            for particle in self.particles:
                particle.draw(self.screen)

            draw_ui(self.screen, self.score, self.lives, self.game_over,
                   self.level, self.show_level_text, self.high_scores if self.game_over else None,
                   self.entering_name, self.current_name,
                   scale=(self.scale_x, self.scale_y))

            pygame.display.flip()

        pygame.quit()
        sys.exit()

    def handle_collisions(self):
        """Handle all game collisions"""
        # Check for bullet-asteroid collisions
        for bullet in self.bullets[:]:
            for asteroid in self.asteroids[:]:
                if asteroid.point_in_asteroid(bullet.pos):
                    try:
                        self.bullets.remove(bullet)
                    except ValueError:
                        pass
                    try:
                        self.asteroids.remove(asteroid)
                    except ValueError:
                        pass
                    self.score += 100

                    # Create explosion particles based on asteroid size
                    if asteroid.size > 30 * self.scale_x:
                        self.sound_effects.play('big_explosion')
                        self.particles.extend(create_explosion_particles(asteroid.pos, 30, (self.scale_x, self.scale_y), 
                                                                      (2, 4), (3, 7)))
                    elif asteroid.size > 15 * self.scale_x:
                        self.sound_effects.play('medium_explosion')
                        self.particles.extend(create_explosion_particles(asteroid.pos, 20, (self.scale_x, self.scale_y), 
                                                                      (1.5, 3), (2.5, 6)))
                    else:
                        self.sound_effects.play('small_explosion')
                        self.particles.extend(create_explosion_particles(asteroid.pos, 15, (self.scale_x, self.scale_y), 
                                                                      (1, 2), (2, 5)))
                    
                    if asteroid.size > 15 * self.scale_x:
                        for _ in range(2):
                            new_size = asteroid.size // 2
                            new_asteroid = Asteroid(asteroid.pos, new_size, scale=(self.scale_x, self.scale_y))
                            new_asteroid.velocity *= asteroid.velocity.length() / 2
                            self.asteroids.append(new_asteroid)
                    break

        # Check for ship-asteroid collisions
        if not self.ship.invulnerable:
            for asteroid in self.asteroids:
                if asteroid.point_in_asteroid(self.ship.pos):
                    self.lives -= 1

                    # Create explosion particles
                    self.particles.extend(create_explosion_particles(self.ship.pos, 30, (self.scale_x, self.scale_y)))
                    life_icon_pos = (self.width - self.lives * 30 * self.scale_x - 30 * self.scale_x, 20 * self.scale_y)
                    self.particles.extend(create_explosion_particles(life_icon_pos, 10, (self.scale_x, self.scale_y)))
                    self.sound_effects.play('ship_explosion')

                    if self.lives > 0:
                        self.ship = self.reset_ship()
                        self.ship.invulnerable = True
                        self.ship.invulnerable_timer = 120
                    else:
                        self.handle_game_over()
                    break

    def handle_game_over(self):
        """Handle transition to game over state"""
        self.in_game = False
        self.game_over = True
        self.sound_effects.stop_all_sounds()
        if self.high_scores.is_high_score(self.score):
            self.entering_name = True
            self.current_name = ""

class Ship:
    def __init__(self, pos, angle=0, scale=(1, 1)):
        self.pos = pygame.Vector2(pos)
        self.angle = angle  # in degrees
        self.velocity = pygame.Vector2(0, 0)
        self.acceleration = 0.5
        self.friction = 0.98
        # Define ship points (nose at top)
        self.points = [pygame.Vector2(0, -15), pygame.Vector2(10, 10), pygame.Vector2(-10, 10)]
        self.invulnerable = False
        self.invulnerable_timer = 0
        self.scale_x, self.scale_y = scale

    def get_nose_position(self):
        """Get the position of the ship's nose for bullet spawning"""
        # Calculate the nose position (15 units up from center, scaled and rotated)
        rad = math.radians(self.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        nose_offset = pygame.Vector2(
            -15 * sin_a * self.scale_x,  # x component
            -15 * cos_a * self.scale_y   # y component
        )
        return self.pos + nose_offset

    def get_rear_position(self):
        """Calculate the position at the rear of the ship"""
        # Calculate the rear position (10 units down from center, scaled and rotated)
        rad = math.radians(self.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        rear_offset = pygame.Vector2(
            10 * sin_a * self.scale_x,  # x component (opposite of nose)
            10 * cos_a * self.scale_y   # y component (opposite of nose)
        )
        return self.pos + rear_offset

    def update(self, thrust, rotate, width, height):
        self.angle += rotate * 5
        
        # Update velocity based on thrust
        if thrust:
            # Use the same angle convention as bullets
            rad = math.radians(self.angle)
            # Thrust in opposite direction of where ship is pointing
            thrust_dir = pygame.Vector2(math.sin(rad), -math.cos(rad))
            self.velocity += thrust_dir * self.acceleration
        
        # Apply friction
        self.velocity *= self.friction
        
        # Update position
        self.pos += self.velocity
        
        # Wrap around screen
        self.pos.x = self.pos.x % width
        self.pos.y = self.pos.y % height
        
        # Update invulnerability
        if self.invulnerable:
            self.invulnerable_timer -= 1
            if self.invulnerable_timer <= 0:
                self.invulnerable = False
        
        # Return thruster particles if thrusting
        if thrust:
            return create_thruster_particles(self.get_rear_position(), self.angle, (self.scale_x, self.scale_y))
        return []

    def make_invulnerable(self, frames=120):  # 2 seconds at 60 FPS
        self.invulnerable = True
        self.invulnerable_timer = frames

    def draw(self, surface):
        if self.invulnerable and pygame.time.get_ticks() % 200 < 100:
            return  # Flash when invulnerable

        # Transform points
        rad = math.radians(self.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        transformed = []
        for point in self.points:
            x = point.x * cos_a - point.y * sin_a
            y = point.x * sin_a + point.y * cos_a
            transformed.append((x * self.scale_x + self.pos.x, y * self.scale_y + self.pos.y))
        pygame.draw.polygon(surface, WHITE, transformed, 1)

class Asteroid:
    def __init__(self, pos, size, scale=(1, 1)):
        self.pos = pygame.Vector2(pos)
        self.size = size  # radius of the asteroid
        self.original_size = size
        angle = random.uniform(0, 360)
        speed = random.uniform(1, 3)
        self.velocity = pygame.Vector2(math.cos(math.radians(angle)), math.sin(math.radians(angle))) * speed
        self.points = self.generate_points()
        self.scale_x, self.scale_y = scale
        # Store the scaled points for collision detection
        self.collision_points = None
        self.update_collision_points()

    def update_collision_points(self):
        """Update the scaled and transformed points for collision detection"""
        self.collision_points = [(x * self.scale_x + self.pos.x, y * self.scale_y + self.pos.y)
                               for x, y in self.points]

    def generate_points(self):
        # Create a rough polygon to represent the asteroid
        num_points = random.randint(8, 12)
        points = []
        for i in range(num_points):
            angle = i * (360 / num_points)
            rad = math.radians(angle)
            # Vary the radius to make it more irregular
            radius = self.size * random.uniform(0.8, 1.2)
            x = math.cos(rad) * radius
            y = math.sin(rad) * radius
            points.append((x, y))
        return points

    def update(self, width, height):
        self.pos += self.velocity
        self.wrap(width, height)
        self.update_collision_points()

    def wrap(self, width, height):
        if self.pos.x > width:
            self.pos.x = 0
        elif self.pos.x < 0:
            self.pos.x = width
        if self.pos.y > height:
            self.pos.y = 0
        elif self.pos.y < 0:
            self.pos.y = height

    def draw(self, surface):
        pygame.draw.polygon(surface, WHITE, self.collision_points, 1)

    def point_in_asteroid(self, point):
        """Check if a point is inside the asteroid using ray casting algorithm"""
        x, y = point
        inside = False

        # Ray casting algorithm
        j = len(self.collision_points) - 1
        for i in range(len(self.collision_points)):
            if ((self.collision_points[i][1] > y) != (self.collision_points[j][1] > y) and
                x < (self.collision_points[j][0] - self.collision_points[i][0]) *
                    (y - self.collision_points[i][1]) /
                    (self.collision_points[j][1] - self.collision_points[i][1]) +
                    self.collision_points[i][0]):
                inside = not inside
            j = i

        return inside

class Bullet:
    def __init__(self, ship_pos, ship_angle, scale=(1, 1)):
        # Start bullet at ship's nose
        rad = math.radians(ship_angle)
        self.pos = pygame.Vector2(ship_pos)
        self.velocity = pygame.Vector2(math.sin(rad), -math.cos(rad)) * 10
        self.lifetime = 60  # frames
        self.scale_x, self.scale_y = scale

    def update(self, width, height):
        self.pos += self.velocity
        self.lifetime -= 1
        self.wrap(width, height)

    def wrap(self, width, height):
        if self.pos.x > width:
            self.pos.x = 0
        elif self.pos.x < 0:
            self.pos.x = width
        if self.pos.y > height:
            self.pos.y = 0
        elif self.pos.y < 0:
            self.pos.y = height

    def draw(self, surface):
        # Draw slightly larger bullet that scales with window size
        radius = max(2 * self.scale_x, 2)
        pygame.draw.circle(surface, RED, (int(self.pos.x), int(self.pos.y)), int(radius))

class Particle:
    def __init__(self, pos, velocity, color, life, size, scale=(1, 1)):
        self.pos = pygame.Vector2(pos)
        self.velocity = pygame.Vector2(velocity)
        self.color = color
        self.life = life
        self.max_life = life
        self.size = size
        self.scale_x, self.scale_y = scale

    def update(self):
        self.pos += self.velocity
        self.life -= 1
        # Fade out the particle
        alpha = int(255 * (self.life / self.max_life))
        self.color = (*self.color[:3], alpha)
        return self.life > 0

    def draw(self, surface):
        # Scale the size based on remaining life
        current_size = self.size * (self.life / self.max_life)
        scaled_size = int(current_size * self.scale_x)
        if scaled_size < 1:
            scaled_size = 1
        
        # Create a surface for the particle with alpha channel
        particle_surface = pygame.Surface((scaled_size * 2, scaled_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(particle_surface, self.color, (scaled_size, scaled_size), scaled_size)
        
        # Draw the particle
        surface.blit(particle_surface, (self.pos.x - scaled_size, self.pos.y - scaled_size))

def create_thruster_particles(pos, angle, scale=(1, 1)):
    particles = []
    # Add 180 degrees to make thrust come out the back
    thrust_angle = angle + 180
    rad = math.radians(thrust_angle)
    thrust_dir = pygame.Vector2(-math.sin(rad), -math.cos(rad))
    
    for _ in range(3):  # Create 3 particles per frame
        spread = random.uniform(-0.5, 0.5)
        # Calculate spread angle
        spread_rad = math.radians(spread * 20)  # 20 degree spread
        cos_spread = math.cos(spread_rad)
        sin_spread = math.sin(spread_rad)
        
        # Rotate the thrust direction by the spread angle
        spread_dir = pygame.Vector2(
            thrust_dir.x * cos_spread - thrust_dir.y * sin_spread,
            thrust_dir.x * sin_spread + thrust_dir.y * cos_spread
        )
        
        speed = random.uniform(3, 6)
        particle_vel = spread_dir * speed
        
        # Start particles at the rear position
        start_pos = pygame.Vector2(pos)
        
        # Orange/red color with random variation
        r = random.randint(200, 255)
        g = random.randint(100, 150)
        b = random.randint(0, 50)
        color = (r, g, b, 255)
        
        life = random.randint(10, 20)
        size = random.uniform(1, 3)
        
        particles.append(Particle(start_pos, particle_vel, color, life, size, scale))
    
    return particles

def create_explosion_particles(pos, num_particles=20, scale=(1, 1), size_range=(1, 3), speed_range=(2, 5), 
                             colors=[(255, 200, 50, 255), (255, 100, 0, 255), (255, 50, 0, 255)]):
    particles = []
    for _ in range(num_particles):
        angle = random.uniform(0, math.pi * 2)
        speed = random.uniform(*speed_range)
        velocity = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
        color = random.choice(colors)
        size = random.uniform(*size_range)
        life = random.randint(20, 40)
        particles.append(Particle(pos, velocity, color, life, size, scale))
    return particles

def draw_ship_icon(surface, pos, scale=(1, 1)):
    # Draw a small ship icon (triangle) at the given position
    points = [pygame.Vector2(0, -8), pygame.Vector2(6, 8), pygame.Vector2(-6, 8)]
    transformed = [(p.x * scale[0] + pos[0], p.y * scale[1] + pos[1]) for p in points]
    pygame.draw.polygon(surface, WHITE, transformed, 0)

class HighScores:
    def __init__(self):
        self.scores = []
        self.load_scores()

    def load_scores(self):
        if HIGH_SCORES_FILE.exists():
            try:
                with open(HIGH_SCORES_FILE, 'r') as f:
                    data = json.load(f)
                    self.scores = data.get('scores', [])
            except (json.JSONDecodeError, IOError):
                self.scores = []
        else:
            self.scores = []

    def save_scores(self):
        with open(HIGH_SCORES_FILE, 'w') as f:
            json.dump({'scores': self.scores}, f)

    def is_high_score(self, score):
        return len(self.scores) < 10 or score > self.scores[-1]['score']

    def add_score(self, name, score):
        self.scores.append({'name': name[:10], 'score': score})  # Limit name to 10 chars
        # Sort scores by score value, highest first
        self.scores.sort(key=lambda x: x['score'], reverse=True)
        # Keep only top 10
        self.scores = self.scores[:10]
        self.save_scores()

def draw_ui(surface, score, lives, game_over, level=1, show_level_text=False, high_scores=None, entering_name=False, current_name="", scale=(1, 1)):
    font = pygame.font.SysFont('Arial', 24)
    # Draw score at top left
    score_text = font.render(f"Score: {score}", True, WHITE)
    surface.blit(score_text, (10, 10))

    # Draw level at top center
    level_text = font.render(f"Level {level}", True, WHITE)
    level_rect = level_text.get_rect(midtop=(surface.get_width() / 2, 10))
    surface.blit(level_text, level_rect)

    # Draw lives icons at upper right
    for i in range(lives):
        icon_x = surface.get_width() - (i + 1) * 30 * scale[0]
        icon_y = 10 * scale[1]
        draw_ship_icon(surface, (icon_x, icon_y + 10 * scale[1]), scale=scale)

    # If showing level announcement
    if show_level_text:
        level_font = pygame.font.SysFont('Arial', 48)
        announce_text = level_font.render(f'Level {level}', True, WHITE)
        text_rect = announce_text.get_rect(center=(surface.get_width() / 2, surface.get_height() / 2))
        surface.blit(announce_text, text_rect)

    # If game over, display messages and high scores
    if game_over:
        y_offset = surface.get_height() / 4

        # Game Over text
        game_font = pygame.font.SysFont('Arial', 48)
        game_over_text = game_font.render('GAME OVER', True, RED)
        text_rect = game_over_text.get_rect(center=(surface.get_width() / 2, y_offset))
        surface.blit(game_over_text, text_rect)
        y_offset += 50

        if entering_name:
            name_prompt = font.render('Enter your name:', True, YELLOW)
            name_rect = name_prompt.get_rect(center=(surface.get_width() / 2, y_offset))
            surface.blit(name_prompt, name_rect)
            y_offset += 30

            name_text = font.render(current_name + "_", True, WHITE)
            name_rect = name_text.get_rect(center=(surface.get_width() / 2, y_offset))
            surface.blit(name_text, name_rect)
            y_offset += 50
        else:
            # Instructions
            restart_text = font.render('Press R to Restart or Q to Quit', True, WHITE)
            restart_rect = restart_text.get_rect(center=(surface.get_width() / 2, y_offset))
            surface.blit(restart_text, restart_rect)
            y_offset += 50

        if high_scores and high_scores.scores:
            # High Scores title
            title_text = game_font.render('High Scores', True, YELLOW)
            title_rect = title_text.get_rect(center=(surface.get_width() / 2, y_offset))
            surface.blit(title_text, title_rect)
            y_offset += 50

            # Display high scores
            for i, score_data in enumerate(high_scores.scores):
                score_text = font.render(f"{i+1}. {score_data['name']:<10} {score_data['score']:>6}", True, WHITE)
                score_rect = score_text.get_rect(center=(surface.get_width() / 2, y_offset))
                surface.blit(score_text, score_rect)
                y_offset += 30

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

        # Thrust sound (rocket engine)
        duration = 1.0
        num_samples = int(duration * sample_rate)
        samples = array.array('h', [0] * (num_samples * 2))

        # Create a more complex rocket engine sound with multiple frequencies and noise
        prev_noise = 0
        for i in range(num_samples):
            t = float(i) / sample_rate

            # Base rumble (40-60 Hz)
            base_rumble = (
                math.sin(2.0 * math.pi * 40.0 * t) * 0.4 +
                math.sin(2.0 * math.pi * 60.0 * t) * 0.3
            )

            # Mid frequencies (120-180 Hz) with slight frequency modulation
            mid_freq = (
                math.sin(2.0 * math.pi * (120.0 + math.sin(t * 2) * 10) * t) * 0.2 +
                math.sin(2.0 * math.pi * (180.0 + math.sin(t * 3) * 15) * t) * 0.15
            )

            # High frequency components (300-500 Hz, quieter)
            high_freq = (
                math.sin(2.0 * math.pi * 300.0 * t) * 0.1 +
                math.sin(2.0 * math.pi * 500.0 * t) * 0.05
            )

            # Add some noise (filtered to be more like air/exhaust)
            noise = random.uniform(-0.3, 0.3)
            # Low-pass filter the noise (simple moving average)
            if i > 0:
                noise = (noise + prev_noise) * 0.5
            prev_noise = noise

            # Combine all components
            value = (base_rumble + mid_freq + high_freq + noise) * max_amplitude * 0.3

            # Add slight stereo effect
            left_value = clamp(int(value * (1.0 + math.sin(t * 2) * 0.1)))
            right_value = clamp(int(value * (1.0 - math.sin(t * 2) * 0.1)))

            samples[i * 2] = left_value
            samples[i * 2 + 1] = right_value

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

    def stop_all_sounds(self):
        """Stop all currently playing sounds"""
        for sound in self.sounds.values():
            sound.stop()
        self.thrust_playing = False

def main():
    game = Game()
    game.run()

if __name__ == "__main__":
    main()
