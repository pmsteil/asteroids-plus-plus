from typing import List, Tuple, Optional, Dict
import pygame
from pygame import Vector2, Surface, mixer
import math
import random
import json
import os
from dataclasses import dataclass

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
ASPECT_RATIO = SCREEN_WIDTH / SCREEN_HEIGHT
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 600

@dataclass
class Particle:
    pos: Vector2
    velocity: Vector2
    color: Tuple[int, int, int, int]
    life: int
    max_life: int
    size: float
    scale_x: float
    scale_y: float

    def update(self) -> bool:
        self.pos += self.velocity
        self.life -= 1
        return self.life > 0

    def draw(self, surface: Surface) -> None:
        alpha = int(255 * (self.life / self.max_life))
        color = (*self.color[:3], alpha)
        size = int(self.size * min(self.scale_x, self.scale_y))
        pygame.draw.circle(surface, color, 
                         (int(self.pos.x), int(self.pos.y)), size)

class Ship:
    def __init__(self, pos: Vector2, angle: float = 0, scale: Tuple[float, float] = (1, 1)):
        self.pos: Vector2 = Vector2(pos)
        self.angle: float = angle
        self.velocity: Vector2 = Vector2(0, 0)
        self.acceleration: float = 0.5
        self.friction: float = 0.98
        self.points: List[Vector2] = [Vector2(0, -15), Vector2(10, 10), Vector2(-10, 10)]
        self.invulnerable: bool = False
        self.invulnerable_timer: int = 0
        self.scale_x, self.scale_y = scale

    def get_nose_position(self) -> Vector2:
        """Calculate the position of the ship's nose for bullet spawning"""
        # Calculate the nose position (15 units up from center, scaled and rotated)
        rad = math.radians(self.angle)
        nose_offset = Vector2(
            15 * math.sin(rad) * self.scale_x,  # x component
            -15 * math.cos(rad) * self.scale_y   # y component
        )
        return self.pos + nose_offset

    def get_rear_position(self) -> Vector2:
        """Calculate the position at the rear of the ship for thrust particles"""
        # Calculate the rear position (opposite of nose position)
        rad = math.radians(self.angle)
        rear_offset = Vector2(
            -15 * math.sin(rad) * self.scale_x,  # x component (opposite of nose)
            15 * math.cos(rad) * self.scale_y    # y component (opposite of nose)
        )
        return self.pos + rear_offset

    def update(self, thrust: bool, rotate: float, width: float, height: float) -> List[Particle]:
        """Update ship position and rotation"""
        # Update rotation
        self.angle += rotate * 5
        self.angle %= 360

        # Update velocity based on thrust
        if thrust:
            # Calculate thrust direction based on ship's angle
            rad = math.radians(self.angle)
            thrust_dir = Vector2(
                math.sin(rad),   # x component
                -math.cos(rad)   # y component
            )
            self.velocity += thrust_dir * self.acceleration

        # Apply friction to slow down
        self.velocity *= self.friction

        # Update position
        self.pos += self.velocity
        
        # Wrap around screen
        self.pos.x %= width
        self.pos.y %= height

        # Update invulnerability
        if self.invulnerable:
            self.invulnerable_timer -= 1
            if self.invulnerable_timer <= 0:
                self.invulnerable = False
        
        if thrust:
            return create_thruster_particles(self.get_rear_position(), self.angle, (self.scale_x, self.scale_y))
        return []

    def make_invulnerable(self, frames: int = 120) -> None:
        self.invulnerable = True
        self.invulnerable_timer = frames

    def draw(self, surface: Surface) -> None:
        if self.invulnerable and self.invulnerable_timer % 2:
            return

        # Transform points based on position, angle, and scale
        rad = math.radians(self.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        transformed_points = []
        for point in self.points:
            x = point.x * cos_a - point.y * sin_a
            y = point.x * sin_a + point.y * cos_a
            transformed_points.append((
                x * self.scale_x + self.pos.x,
                y * self.scale_y + self.pos.y
            ))
        
        pygame.draw.polygon(surface, (255, 255, 255), transformed_points, 2)

class Bullet:
    def __init__(self, ship_pos: Vector2, ship_angle: float, scale: Tuple[float, float] = (1, 1)):
        self.pos: Vector2 = Vector2(ship_pos)
        rad = math.radians(ship_angle)
        self.velocity: Vector2 = Vector2(math.sin(rad), -math.cos(rad)) * 10
        self.lifetime: int = 60
        self.scale_x, self.scale_y = scale

    def update(self, width: int, height: int) -> bool:
        self.pos += self.velocity
        self.lifetime -= 1
        self.wrap(width, height)
        return self.lifetime > 0

    def wrap(self, width: int, height: int) -> None:
        self.pos.x = self.pos.x % width
        self.pos.y = self.pos.y % height

    def draw(self, surface: Surface) -> None:
        size = int(2 * min(self.scale_x, self.scale_y))
        pygame.draw.circle(surface, (255, 255, 255), 
                         (int(self.pos.x), int(self.pos.y)), size)

class Asteroid:
    def __init__(self, pos: Vector2, size: float, scale: Tuple[float, float] = (1, 1)):
        self.pos: Vector2 = Vector2(pos)
        self.size: float = size
        self.original_size: float = size
        angle = random.uniform(0, 360)
        speed = random.uniform(1, 3)
        self.velocity: Vector2 = Vector2(
            math.cos(math.radians(angle)),
            math.sin(math.radians(angle))
        ) * speed
        self.points: List[Vector2] = self.generate_points()
        self.scale_x, self.scale_y = scale
        self.collision_points: Optional[List[Tuple[float, float]]] = None
        self.update_collision_points()

    def generate_points(self) -> List[Vector2]:
        num_points = random.randint(8, 12)
        points = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            distance = self.size * random.uniform(0.8, 1.2)
            points.append(Vector2(
                math.cos(angle) * distance,
                math.sin(angle) * distance
            ))
        return points

    def update_collision_points(self) -> None:
        rad = 0  # Asteroids don't rotate yet
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        self.collision_points = []
        for point in self.points:
            x = point.x * cos_a - point.y * sin_a
            y = point.x * sin_a + point.y * cos_a
            self.collision_points.append((
                x * self.scale_x + self.pos.x,
                y * self.scale_y + self.pos.y
            ))

    def update(self, width: int, height: int) -> None:
        self.pos += self.velocity
        self.wrap(width, height)
        self.update_collision_points()

    def wrap(self, width: int, height: int) -> None:
        self.pos.x = self.pos.x % width
        self.pos.y = self.pos.y % height

    def draw(self, surface: Surface) -> None:
        if self.collision_points:
            pygame.draw.polygon(surface, (255, 255, 255), self.collision_points, 2)

    def point_in_asteroid(self, point: Tuple[float, float]) -> bool:
        if not self.collision_points:
            return False
            
        x, y = point
        inside = False
        j = len(self.collision_points) - 1
        
        for i in range(len(self.collision_points)):
            xi, yi = self.collision_points[i]
            xj, yj = self.collision_points[j]
            
            if ((yi > y) != (yj > y) and
                x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
            
        return inside

class SoundEffects:
    def __init__(self) -> None:
        self.sounds: Dict[str, mixer.Sound] = {}
        self._create_sounds()
        self.thrust_playing: bool = False
        self.beat_tempo: float = 1.0
        self.last_beat_time: int = 0

    def _create_sounds(self) -> None:
        import array
        sample_rate = 44100
        max_amplitude = 32767  # Max value for 16-bit audio

        def clamp(value: int) -> int:
            """Clamp value to valid 16-bit range"""
            return max(min(value, 32767), -32768)

        def create_buffer(samples: array.array) -> array.array:
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
        self.sounds['fire'] = mixer.Sound(buffer=samples)
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
            value = clamp(int(max_amplitude * (base_rumble + mid_freq + high_freq + noise)))
            samples[i * 2] = value  # Left channel
            samples[i * 2 + 1] = value  # Right channel

        self.sounds['thrust'] = mixer.Sound(buffer=samples)
        self.sounds['thrust'].set_volume(0.4)

        # Explosion sounds
        for size in ['large', 'medium', 'small']:
            duration = 1.0 if size == 'large' else 0.7 if size == 'medium' else 0.5
            base_freq = 100 if size == 'large' else 200 if size == 'medium' else 300
            volume = 0.7 if size == 'large' else 0.5 if size == 'medium' else 0.3
            
            num_samples = int(duration * sample_rate)
            samples = array.array('h', [0] * (num_samples * 2))
            
            for i in range(num_samples):
                t = float(i) / sample_rate
                time_factor = 1.0 - (t / duration)  # Linear decay
                
                # Base explosion sound
                base = math.sin(2.0 * math.pi * base_freq * t) * 0.5
                
                # Add noise that gets quieter over time
                noise = random.uniform(-0.5, 0.5) * time_factor
                
                # Combine and apply volume envelope
                value = clamp(int(max_amplitude * (base + noise) * time_factor * volume))
                samples[i * 2] = value
                samples[i * 2 + 1] = value
                
            self.sounds[f'explosion_{size}'] = mixer.Sound(buffer=samples)

    def play(self, sound_name: str) -> None:
        if sound_name in self.sounds:
            self.sounds[sound_name].play()

    def start_thrust(self) -> None:
        if not self.thrust_playing:
            self.sounds['thrust'].play(-1)  # Loop indefinitely
            self.thrust_playing = True

    def stop_thrust(self) -> None:
        if self.thrust_playing:
            self.sounds['thrust'].stop()
            self.thrust_playing = False

    def update_beat(self, asteroid_count: int) -> None:
        # Increase tempo based on fewer asteroids
        self.beat_tempo = 1.0 + max(0, (10 - asteroid_count) / 10)
        
        current_time = pygame.time.get_ticks()
        if current_time - self.last_beat_time > 1000 / self.beat_tempo:
            self.play('beat')
            self.last_beat_time = current_time

    def stop_all_sounds(self) -> None:
        for sound in self.sounds.values():
            sound.stop()
        self.thrust_playing = False

def create_thruster_particles(pos: Vector2, angle: float, scale: Tuple[float, float] = (1, 1)) -> List[Particle]:
    particles = []
    rad = math.radians(angle)
    thrust_dir = Vector2(math.sin(rad), -math.cos(rad))
    
    for _ in range(3):
        spread = random.uniform(-0.5, 0.5)
        spread_rad = math.radians(spread * 20)
        cos_spread = math.cos(spread_rad)
        sin_spread = math.sin(spread_rad)
        
        spread_dir = Vector2(
            thrust_dir.x * cos_spread - thrust_dir.y * sin_spread,
            thrust_dir.x * sin_spread + thrust_dir.y * cos_spread
        )
        
        speed = random.uniform(3, 6)
        particle_vel = -spread_dir * speed
        
        start_pos = Vector2(pos)
        
        r = random.randint(200, 255)
        g = random.randint(100, 150)
        b = random.randint(0, 50)
        color = (r, g, b, 255)
        
        life = random.randint(10, 20)
        size = random.uniform(1, 3)
        
        particles.append(Particle(start_pos, particle_vel, color, life, life, size, scale[0], scale[1]))
    
    return particles

def create_explosion_particles(
    pos: Vector2,
    num_particles: int = 20,
    scale: Tuple[float, float] = (1, 1),
    size_range: Tuple[float, float] = (1, 3),
    speed_range: Tuple[float, float] = (2, 5),
    colors: List[Tuple[int, int, int, int]] = [(255, 200, 50, 255), (255, 100, 0, 255), (255, 50, 0, 255)]
) -> List[Particle]:
    particles = []
    for _ in range(num_particles):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(*speed_range)
        velocity = Vector2(math.cos(angle), math.sin(angle)) * speed
        color = random.choice(colors)
        life = random.randint(20, 40)
        size = random.uniform(*size_range)
        particles.append(Particle(Vector2(pos), velocity, color, life, life, size, scale[0], scale[1]))
    return particles

def draw_ship_icon(surface: Surface, pos: Vector2, scale: Tuple[float, float] = (1, 1)) -> None:
    points = [
        Vector2(0, -10),
        Vector2(7, 7),
        Vector2(-7, 7)
    ]
    transformed_points = [
        (p.x * scale[0] + pos.x, p.y * scale[1] + pos.y)
        for p in points
    ]
    pygame.draw.polygon(surface, (255, 255, 255), transformed_points, 2)

def draw_ui(
    surface: Surface,
    score: int,
    lives: int,
    game_over: bool,
    level: int = 1,
    show_level_text: bool = False,
    high_scores: Optional[List[Tuple[str, int]]] = None,
    entering_name: bool = False,
    current_name: str = "",
    scale: Tuple[float, float] = (1, 1)
) -> None:
    font = pygame.font.SysFont('Arial', 24)
    
    # Score at top left
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    surface.blit(score_text, (10, 10))
    
    # Level at top center
    level_text = font.render(f"Level {level}", True, (255, 255, 255))
    level_rect = level_text.get_rect(midtop=(surface.get_width() / 2, 10))
    surface.blit(level_text, level_rect)
    
    # Lives icons at upper right
    for i in range(lives):
        icon_x = surface.get_width() - (i + 1) * 30 * scale[0]
        icon_y = 10 * scale[1]
        draw_ship_icon(surface, Vector2(icon_x, icon_y + 10 * scale[1]), scale)
    
    # Level announcement
    if show_level_text:
        level_font = pygame.font.SysFont('Arial', 48)
        announce_text = level_font.render(f'Level {level}', True, (255, 255, 255))
        text_rect = announce_text.get_rect(center=(surface.get_width() / 2, surface.get_height() / 2))
        surface.blit(announce_text, text_rect)
    
    # Game over screen
    if game_over:
        y_offset = surface.get_height() / 4
        
        game_font = pygame.font.SysFont('Arial', 48)
        game_over_text = game_font.render('GAME OVER', True, (255, 0, 0))
        text_rect = game_over_text.get_rect(center=(surface.get_width() / 2, y_offset))
        surface.blit(game_over_text, text_rect)
        y_offset += 50
        
        if entering_name:
            name_text = font.render(f"Enter name: {current_name}_", True, (255, 255, 255))
            text_rect = name_text.get_rect(center=(surface.get_width() / 2, y_offset))
            surface.blit(name_text, text_rect)
            y_offset += 50
        
        if high_scores:
            title_text = font.render("High Scores:", True, (255, 255, 255))
            text_rect = title_text.get_rect(center=(surface.get_width() / 2, y_offset))
            surface.blit(title_text, text_rect)
            y_offset += 30
            
            for name, score in high_scores[:5]:
                score_text = font.render(f"{name}: {score}", True, (255, 255, 255))
                text_rect = score_text.get_rect(center=(surface.get_width() / 2, y_offset))
                surface.blit(score_text, text_rect)
                y_offset += 30

class Game:
    def __init__(self) -> None:
        pygame.init()
        mixer.init(44100, -16, 2, 1024)
        
        # Get display info
        info = pygame.display.Info()
        self.max_width = info.current_w
        self.max_height = info.current_h
        
        # Set initial window size
        target_width = int(self.max_width * 0.8)
        target_height = int(target_width / ASPECT_RATIO)
        if target_height > self.max_height * 0.8:
            target_height = int(self.max_height * 0.8)
            target_width = int(target_height * ASPECT_RATIO)
        
        self.width = target_width
        self.height = target_height
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("Asteroids")
        
        # Game objects
        self.scale_x = self.width / DEFAULT_WIDTH
        self.scale_y = self.height / DEFAULT_HEIGHT
        self.ship_collision_radius = 15 * self.scale_x
        self.clock = pygame.time.Clock()
        self.sound_effects = SoundEffects()
        
        # Game state
        self.reset_game_state()

    def reset_game_state(self) -> None:
        self.score = 0
        self.lives = 3
        self.level = 1
        self.game_over = False
        self.show_level_text = True
        self.level_text_timer = 120
        self.entering_name = False
        self.current_name = ""
        self.reset_ship()
        self.asteroids = []
        self.bullets = []
        self.particles = []
        self.start_new_level(self.level)

    def reset_ship(self) -> None:
        self.ship = Ship(Vector2(self.width / 2, self.height / 2), 0, (self.scale_x, self.scale_y))
        self.ship.make_invulnerable()

    def handle_resize(self, new_width: int, new_height: int) -> None:
        self.width = new_width
        self.height = new_height
        self.scale_x = self.width / DEFAULT_WIDTH
        self.scale_y = self.height / DEFAULT_HEIGHT
        self.ship_collision_radius = 15 * self.scale_x
        
        # Update existing objects with new scale
        if self.ship:
            self.ship.scale_x, self.ship.scale_y = self.scale_x, self.scale_y
        for asteroid in self.asteroids:
            asteroid.scale_x, asteroid.scale_y = self.scale_x, self.scale_y
            asteroid.update_collision_points()
        for bullet in self.bullets:
            bullet.scale_x, bullet.scale_y = self.scale_x, self.scale_y
        for particle in self.particles:
            particle.scale_x, particle.scale_y = self.scale_x, self.scale_y

    def start_new_level(self, level_num: int) -> None:
        self.level = level_num
        num_asteroids = 3 + (level_num - 1)
        
        for _ in range(num_asteroids):
            # Spawn asteroids away from the ship
            while True:
                x = random.randrange(self.width)
                y = random.randrange(self.height)
                if Vector2(x - self.ship.pos.x, y - self.ship.pos.y).length() > 100:
                    break
            
            self.asteroids.append(
                Asteroid(Vector2(x, y), 40, (self.scale_x, self.scale_y))
            )
        
        self.show_level_text = True
        self.level_text_timer = 120

    def handle_collisions(self) -> None:
        # Bullet-asteroid collisions
        for bullet in self.bullets[:]:
            for asteroid in self.asteroids[:]:
                if asteroid.point_in_asteroid((bullet.pos.x, bullet.pos.y)):
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                    if asteroid in self.asteroids:
                        self.asteroids.remove(asteroid)
                        
                        # Add explosion particles
                        self.particles.extend(
                            create_explosion_particles(asteroid.pos, 20,
                                                    (self.scale_x, self.scale_y))
                        )
                        
                        # Play explosion sound based on size
                        if asteroid.size >= 30:
                            self.sound_effects.play('explosion_large')
                            self.score += 20
                        elif asteroid.size >= 15:
                            self.sound_effects.play('explosion_medium')
                            self.score += 50
                        else:
                            self.sound_effects.play('explosion_small')
                            self.score += 100
                        
                        # Split asteroid if large enough
                        if asteroid.size >= 20:
                            for _ in range(2):
                                new_asteroid = Asteroid(
                                    Vector2(asteroid.pos),
                                    asteroid.size / 2,
                                    (self.scale_x, self.scale_y)
                                )
                                self.asteroids.append(new_asteroid)
        
        # Ship-asteroid collisions
        if self.ship and not self.ship.invulnerable:
            ship_pos = (self.ship.pos.x, self.ship.pos.y)
            for asteroid in self.asteroids:
                if asteroid.point_in_asteroid(ship_pos):
                    self.lives -= 1
                    self.sound_effects.play('explosion_medium')
                    self.particles.extend(
                        create_explosion_particles(self.ship.pos, 30,
                                                (self.scale_x, self.scale_y))
                    )
                    if self.lives > 0:
                        self.reset_ship()
                    else:
                        self.ship = None
                        self.handle_game_over()
                    break

    def handle_game_over(self) -> None:
        self.game_over = True
        self.sound_effects.stop_all_sounds()

    def run(self) -> None:
        running = True
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    self.handle_resize(event.w, event.h)
                elif event.type == pygame.KEYDOWN and self.game_over and self.entering_name:
                    if event.key == pygame.K_RETURN and self.current_name:
                        self.high_scores.add_score(self.current_name, self.score)
                        self.entering_name = False
                    elif event.key == pygame.K_BACKSPACE:
                        self.current_name = self.current_name[:-1]
                    elif len(self.current_name) < 10 and event.unicode.isalnum():
                        self.current_name += event.unicode
            
            # Get input
            keys = pygame.key.get_pressed()
            if not self.game_over and self.ship:
                # Rotation
                rotate = 0
                if keys[pygame.K_LEFT]:
                    rotate = -1
                elif keys[pygame.K_RIGHT]:
                    rotate = 1
                
                # Thrust
                thrust = keys[pygame.K_UP]
                if thrust:
                    self.sound_effects.start_thrust()
                else:
                    self.sound_effects.stop_thrust()
                
                # Fire
                if keys[pygame.K_SPACE]:
                    if not hasattr(self, 'last_fire_time'):
                        self.last_fire_time = 0
                    current_time = pygame.time.get_ticks()
                    if current_time - self.last_fire_time > 250:  # Fire rate limit
                        bullet_pos = self.ship.get_nose_position()
                        rad = math.radians(self.ship.angle)
                        bullet_vel = Vector2(math.sin(rad), -math.cos(rad)) * 10
                        self.bullets.append(
                            Bullet(bullet_pos, self.ship.angle, (self.scale_x, self.scale_y))
                        )
                        self.bullets[-1].velocity = bullet_vel
                        self.sound_effects.play('fire')
                        self.last_fire_time = current_time
                
                # Update ship and create particles
                new_particles = self.ship.update(thrust, rotate, self.width, self.height)
                self.particles.extend(new_particles)
            
            # Update game objects
            self.bullets = [b for b in self.bullets if b.update(self.width, self.height)]
            for asteroid in self.asteroids:
                asteroid.update(self.width, self.height)
            self.particles = [p for p in self.particles if p.update()]
            
            # Handle collisions
            self.handle_collisions()
            
            # Check for level completion
            if not self.game_over and not self.asteroids:
                self.start_new_level(self.level + 1)
            
            # Update level text timer
            if self.show_level_text:
                self.level_text_timer -= 1
                if self.level_text_timer <= 0:
                    self.show_level_text = False
            
            # Draw
            self.screen.fill((0, 0, 0))
            
            if self.ship:
                self.ship.draw(self.screen)
            for bullet in self.bullets:
                bullet.draw(self.screen)
            for asteroid in self.asteroids:
                asteroid.draw(self.screen)
            for particle in self.particles:
                particle.draw(self.screen)
            
            # Draw UI
            draw_ui(self.screen, self.score, self.lives, self.game_over,
                   self.level, self.show_level_text, None,
                   self.entering_name, self.current_name,
                   (self.scale_x, self.scale_y))
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    game = Game()
    game.run()
