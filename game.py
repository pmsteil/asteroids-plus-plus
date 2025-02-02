from typing import List, Tuple, Optional, Dict, Union
import pygame
from pygame import mixer
from pygame.surface import Surface
import sys
import random
import math
import json
import os
import array
from dataclasses import dataclass
from pygame.math import Vector2

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
ASPECT_RATIO = SCREEN_WIDTH / SCREEN_HEIGHT
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 600
HIGH_SCORES_FILE = os.path.join(os.path.dirname(__file__), "high_scores.json")
EXTRA_LIFE_SCORE = 10000  # Score needed for an extra life
STARTING_ASTEROIDS = 25  # Number of asteroids at level 1
ASTEROIDS_LEVEL_INCREASE = 25  # Percentage increase in asteroids per level
STARTING_GUNS = 4  # Number of guns to start with (1-4)
ASTEROID_SIZES = {
    'LARGE': 40,
    'MEDIUM': 25,
    'SMALL': 15
}

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
    def __init__(self, pos: Vector2, scale: Tuple[float, float] = (1, 1)) -> None:
        """Initialize the ship"""
        self.pos = Vector2(pos)
        self.velocity = Vector2(0, 0)
        self.angle = 0  # 0 is pointing up, increases clockwise
        self.scale_x, self.scale_y = scale
        self.invulnerable = True
        self.invulnerable_timer = 120  # 2 seconds at 60 FPS

        # Define ship shape relative to center
        self.points = [
            Vector2(0, -10),  # Nose
            Vector2(6, 10),   # Bottom right
            Vector2(0, 7),    # Bottom middle
            Vector2(-6, 10)   # Bottom left
        ]

    def get_nose_position(self) -> Vector2:
        """Get the position of the ship's nose in world coordinates"""
        # Convert angle to radians
        rad = math.radians(self.angle)

        # Get the nose point (first point in self.points)
        nose = self.points[0]

        # Rotate the nose point
        rotated_x = nose.x * math.cos(rad) - nose.y * math.sin(rad)
        rotated_y = nose.x * math.sin(rad) + nose.y * math.cos(rad)

        # Scale and translate to world position
        world_pos = Vector2(
            self.pos.x + rotated_x * self.scale_x,
            self.pos.y + rotated_y * self.scale_y
        )

        return world_pos

    def get_rear_position(self) -> Vector2:
        """Get the position of the ship's rear center in world coordinates"""
        # Convert angle to radians
        rad = math.radians(self.angle)

        # Get the rear center point (third point in self.points)
        rear = self.points[2]

        # Rotate the rear point
        rotated_x = rear.x * math.cos(rad) - rear.y * math.sin(rad)
        rotated_y = rear.x * math.sin(rad) + rear.y * math.cos(rad)

        # Scale and translate to world position
        world_pos = Vector2(
            self.pos.x + rotated_x * self.scale_x,
            self.pos.y + rotated_y * self.scale_y
        )

        return world_pos

    def get_transformed_points(self) -> List[Vector2]:
        """Get the ship's points transformed by position, rotation and scale"""
        rad = math.radians(self.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        transformed = []
        for point in self.points:
            # Scale and rotate the point
            x = point.x * self.scale_x
            y = point.y * self.scale_y
            rotated_x = x * cos_a - y * sin_a
            rotated_y = x * sin_a + y * cos_a
            # Translate to ship's position
            transformed.append(Vector2(
                rotated_x + self.pos.x,
                rotated_y + self.pos.y
            ))
        return transformed

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
            self.velocity += thrust_dir * 0.5

        # Apply friction to slow down
        self.velocity *= 0.98

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

    def make_invulnerable(self, frames: int = 240) -> None:
        """Make ship invulnerable for the specified number of frames"""
        self.invulnerable = True
        self.invulnerable_timer = frames

    def draw(self, surface: Surface) -> None:
        """Draw the ship"""
        # Transform points
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

        if self.invulnerable:
            # Create smooth pulse using sine wave
            # 30 frames for one complete pulse (2 seconds)
            pulse = (math.sin(self.invulnerable_timer * math.pi / 30) + 1) / 2  # Range 0 to 1
            # Keep minimum visibility at 40% and max at 100%
            pulse = 0.4 + (pulse * 0.6)  # Range 0.4 to 1.0
            # Create muted green that pulses but stays visible
            color = (int(40 * pulse), int(180 * pulse), int(40 * pulse))
        else:
            color = (255, 255, 255)  # Normal white color

        pygame.draw.polygon(surface, color, transformed_points, 2)

class Bullet:
    def __init__(self, pos: Vector2, velocity: Vector2, scale: Tuple[float, float]) -> None:
        """Initialize a bullet"""
        self.pos = Vector2(pos)
        self.velocity = Vector2(velocity)
        self.lifetime = 60
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
        self.collision_points: Optional[List[Vector2]] = None
        self.update_collision_points()

    def generate_points(self) -> List[Vector2]:
        """Generate the asteroid's shape points"""
        points = []
        num_points = random.randint(8, 12)
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            # Vary the radius to make the asteroid more irregular
            radius = self.size * random.uniform(0.8, 1.2)
            points.append(Vector2(
                radius * math.cos(angle),
                radius * math.sin(angle)
            ))
        return points

    def update_collision_points(self) -> None:
        """Update the scaled and transformed points for collision detection"""
        transformed = []
        for point in self.points:
            transformed.append(Vector2(
                point.x * self.scale_x + self.pos.x,
                point.y * self.scale_y + self.pos.y
            ))
        self.collision_points = transformed

    def update(self, width: int, height: int) -> None:
        self.pos += self.velocity
        self.wrap(width, height)
        self.update_collision_points()

    def wrap(self, width: int, height: int) -> None:
        self.pos.x = self.pos.x % width
        self.pos.y = self.pos.y % height

    def draw(self, surface: Surface) -> None:
        """Draw the asteroid"""
        # Draw the asteroid shape
        if self.collision_points:
            points = [(p.x, p.y) for p in self.collision_points]
            pygame.draw.polygon(surface, (255, 255, 255), points, 2)

    def point_in_asteroid(self, point: Vector2) -> bool:
        """Check if a point is inside the asteroid using ray casting"""
        # Get the transformed points for collision detection
        points = self.collision_points
        if not points:
            return False

        # Ray casting algorithm
        inside = False
        j = len(points) - 1

        for i in range(len(points)):
            if (((points[i].y > point.y) != (points[j].y > point.y)) and
                (point.x < (points[j].x - points[i].x) * (point.y - points[i].y) /
                          (points[j].y - points[i].y) + points[i].x)):
                inside = not inside
            j = i

        return inside

    def line_segments_intersect(self, p1: Vector2, p2: Vector2, p3: Vector2, p4: Vector2) -> bool:
        """Check if line segments (p1,p2) and (p3,p4) intersect"""
        def ccw(A: Vector2, B: Vector2, C: Vector2) -> bool:
            """Returns True if points are arranged counter-clockwise"""
            return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)

        # Check if line segments intersect using CCW tests
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def check_collision_with_ship(self, ship: Ship) -> bool:
        """Check for collision between asteroid and ship"""
        ship_points = ship.get_transformed_points()
        asteroid_points = self.collision_points

        if not asteroid_points:
            return False

        # First check if any ship point is inside the asteroid
        for point in ship_points:
            if self.point_in_asteroid(point):
                return True

        # Then check if any asteroid point is inside the ship polygon
        for point in asteroid_points:
            inside = False
            j = len(ship_points) - 1
            for i in range(len(ship_points)):
                if (((ship_points[i].y > point.y) != (ship_points[j].y > point.y)) and
                    (point.x < (ship_points[j].x - ship_points[i].x) * (point.y - ship_points[i].y) /
                              (ship_points[j].y - ship_points[i].y) + ship_points[i].x)):
                    inside = not inside
                j = i
            if inside:
                return True

        # Finally check for line segment intersections
        for i in range(len(ship_points)):
            p1 = ship_points[i]
            p2 = ship_points[(i + 1) % len(ship_points)]

            for j in range(len(asteroid_points)):
                p3 = asteroid_points[j]
                p4 = asteroid_points[(j + 1) % len(asteroid_points)]

                if self.line_segments_intersect(p1, p2, p3, p4):
                    return True

        return False

class SoundEffects:
    def __init__(self) -> None:
        self.sounds: Dict[str, mixer.Sound] = {}
        self._create_sounds()
        self.thrust_playing: bool = False
        self.beat_tempo: float = 1.0
        self.last_beat_time: int = 0

    def create_final_explosion(self) -> mixer.Sound:
        """Create a longer, more dramatic explosion sound for game over"""
        sample_rate = 44100
        duration = 3.0  # 3 seconds
        num_samples = int(duration * sample_rate)
        samples = array.array('h', [0] * (num_samples * 2))

        for i in range(num_samples):
            t = float(i) / sample_rate
            time_factor = 1.0 - (t / duration)  # Linear decay

            # Base explosion (50-150 Hz, sweeping down)
            base_freq = 150 - 100 * (t / duration)
            base = math.sin(2.0 * math.pi * base_freq * t) * 0.6

            # Mid frequencies (200-400 Hz)
            mid = (
                math.sin(2.0 * math.pi * 200 * t) * 0.3 +
                math.sin(2.0 * math.pi * 400 * t) * 0.2
            )

            # High frequency components (300-500 Hz, quieter)
            high_freq = (
                math.sin(2.0 * math.pi * 300 * t) * 0.1 +
                math.sin(2.0 * math.pi * 500 * t) * 0.05
            )

            # Add some noise (filtered to be more like air/exhaust)
            noise = random.uniform(-0.4, 0.4) * time_factor

            # Combine all components with time-based envelope
            value = int(32767 * (base + mid + high_freq + noise) * time_factor)
            value = max(min(value, 32767), -32768)

            samples[i * 2] = value  # Left channel
            samples[i * 2 + 1] = value  # Right channel

        return mixer.Sound(buffer=samples)

    def _create_power_up_sound(self) -> mixer.Sound:
        """Create a power up sound for extra life"""
        sample_rate = 44100
        duration = 1.0
        num_samples = int(duration * sample_rate)
        samples = array.array('h', [0] * (num_samples * 2))

        for i in range(num_samples):
            t = float(i) / sample_rate

            # Rising pitch from 220Hz to 880Hz (A3 to A5)
            freq = 220 + (660 * (t / duration))

            # Amplitude envelope: quick attack, long decay
            envelope = min(1.0, t * 10) * (1.0 - t)

            # Main tone plus harmonics
            value = (
                math.sin(2.0 * math.pi * freq * t) * 0.5 +  # Base frequency
                math.sin(2.0 * math.pi * freq * 2 * t) * 0.25 +  # First harmonic
                math.sin(2.0 * math.pi * freq * 4 * t) * 0.125  # Second harmonic
            )

            # Apply envelope
            value = int(32767 * value * envelope)
            value = max(min(value, 32767), -32768)

            samples[i * 2] = value
            samples[i * 2 + 1] = value

        return mixer.Sound(buffer=samples)

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
                value = max(min(value, 32767), -32768)

                samples[i * 2] = value
                samples[i * 2 + 1] = value

            self.sounds[f'explosion_{size}'] = mixer.Sound(buffer=samples)

        # Final explosion sound
        self.sounds['final_explosion'] = self.create_final_explosion()

        # Power up sound
        self.sounds['power_up'] = self._create_power_up_sound()

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

class HighScores:
    def __init__(self) -> None:
        self.scores: List[Dict[str, Union[str, int]]] = []
        self.load_scores()

    def load_scores(self) -> None:
        """Load high scores from file or create if doesn't exist"""
        try:
            with open(HIGH_SCORES_FILE, 'r') as f:
                data = json.load(f)
                self.scores = data.get('scores', [])
                # Add level field if it doesn't exist in older scores
                for score in self.scores:
                    if 'level' not in score:
                        score['level'] = 1
        except (FileNotFoundError, json.JSONDecodeError):
            # Create new empty scores file
            self.scores = []
            self.save_scores()

    def save_scores(self) -> None:
        """Save high scores to file"""
        with open(HIGH_SCORES_FILE, 'w') as f:
            json.dump({'scores': self.scores}, f)

    def is_high_score(self, score: int) -> bool:
        """Check if score qualifies for high scores list"""
        return len(self.scores) < 10 or score > self.scores[-1]['score']

    def add_score(self, name: str, score: int, level: int) -> None:
        """Add a new high score with level reached"""
        self.scores.append({
            'name': name[:10],  # Limit name to 10 chars
            'score': score,
            'level': level
        })
        # Sort scores by score value, highest first
        self.scores.sort(key=lambda x: x['score'], reverse=True)
        # Keep only top 10
        self.scores = self.scores[:10]
        self.save_scores()

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

def draw_ship_icon(surface: Surface, pos: Union[Vector2, Tuple[float, float]],
                  scale: Tuple[float, float] = (1, 1),
                  color: Tuple[int, int, int] = (255, 255, 255)) -> None:
    """Draw a small ship icon for the lives display"""
    # Convert tuple to Vector2 if needed
    if isinstance(pos, tuple):
        pos = Vector2(pos)

    # Define ship points (smaller version of ship)
    points = [
        Vector2(0, -8),  # Nose
        Vector2(5, 5),   # Right
        Vector2(-5, 5)   # Left
    ]

    # Scale and transform points
    screen_points = []
    for p in points:
        screen_points.append(
            (p.x * scale[0] + pos.x, p.y * scale[1] + pos.y)
        )

    # Draw the ship outline
    pygame.draw.polygon(surface, color, screen_points, 1)

def draw_ui(surface: Surface, score: int, lives: int, level: int, scale: Tuple[float, float],
            game_over: bool = False, show_level_text: bool = False,
            entering_name: bool = False, current_name: str = "",
            high_scores: Optional[HighScores] = None,
            new_life_timer: int = 0) -> None:
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)
    GRAY = (180, 180, 180)

    font = pygame.font.SysFont('Arial', int(24 * scale[0]))
    small_font = pygame.font.SysFont('Arial', int(14 * scale[0]))

    # Draw score at top left
    score_text = font.render(f"Score: {score}", True, WHITE)
    surface.blit(score_text, (10 * scale[0], 10 * scale[1]))

    # Only draw level at top if not showing level announcement
    if not show_level_text:
        level_text = font.render(f"Level {level}", True, WHITE)
        level_rect = level_text.get_rect(midtop=(surface.get_width() / 2, 10 * scale[1]))
        surface.blit(level_text, level_rect)

    # Draw lives at upper right
    lives_x = surface.get_width() - (35 * scale[0])
    lives_y = 20 * scale[1]

    for i in range(lives):
        # If this is the newest life and animation is active, pulse it
        if i == lives - 1 and new_life_timer > 0:
            # Use the same pulse logic as ship invulnerability but slower
            pulse = (math.sin(new_life_timer * math.pi / 30) + 1) / 2  # Slower pulse
            pulse = 0.4 + (pulse * 0.6)  # Range 0.4 to 1.0
            color = (int(40 * pulse), int(180 * pulse), int(40 * pulse))
        else:
            color = WHITE

        draw_ship_icon(surface, (lives_x - i * 30 * scale[0], lives_y), scale=scale, color=color)

    # If showing level announcement
    if show_level_text:
        level_font = pygame.font.SysFont('Arial', int(48 * scale[0]))
        announce_text = level_font.render(f'Level {level}', True, WHITE)
        text_rect = announce_text.get_rect(center=(surface.get_width() / 2, surface.get_height() / 2))
        surface.blit(announce_text, text_rect)

    # If game over, display messages and high scores
    if game_over:
        y_offset = surface.get_height() / 4

        # Game Over text
        game_font = pygame.font.SysFont('Arial', int(48 * scale[0]))
        game_over_text = game_font.render('GAME OVER', True, RED)
        text_rect = game_over_text.get_rect(center=(surface.get_width() / 2, y_offset))
        surface.blit(game_over_text, text_rect)
        y_offset += 50 * scale[1]

        if entering_name:
            name_prompt = font.render('Enter your name:', True, YELLOW)
            name_rect = name_prompt.get_rect(center=(surface.get_width() / 2, y_offset))
            surface.blit(name_prompt, name_rect)
            y_offset += 30 * scale[1]

            name_text = font.render(current_name + "_", True, WHITE)
            name_rect = name_text.get_rect(center=(surface.get_width() / 2, y_offset))
            surface.blit(name_text, name_rect)
            y_offset += 50 * scale[1]
        else:
            # Instructions
            restart_text = font.render('Press R to Restart or Q to Quit', True, WHITE)
            restart_rect = restart_text.get_rect(center=(surface.get_width() / 2, y_offset))
            surface.blit(restart_text, restart_rect)

            y_offset += 50 * scale[1]

        if high_scores and high_scores.scores:
            # High Scores title
            title_text = game_font.render('High Scores', True, YELLOW)
            title_rect = title_text.get_rect(center=(surface.get_width() / 2, y_offset))
            surface.blit(title_text, title_rect)
            y_offset += 40 * scale[1]  # Reduced spacing

            # Column headers
            header_text = small_font.render("Rank  Name         Score   Level", True, GRAY)
            header_rect = header_text.get_rect(center=(surface.get_width() / 2, y_offset))
            surface.blit(header_text, header_rect)
            y_offset += 25 * scale[1]  # Reduced spacing

            # Display high scores
            for i, score_data in enumerate(high_scores.scores):
                level_text = f"L{score_data.get('level', 1)}"
                score_text = small_font.render(
                    f"{i+1:2d}.  {score_data['name']:<10} {score_data['score']:>6}  {level_text:>3}",
                    True,
                    WHITE
                )
                score_rect = score_text.get_rect(center=(surface.get_width() / 2, y_offset))
                surface.blit(score_text, score_rect)
                y_offset += 25 * scale[1]  # Reduced spacing

class Game:
    def __init__(self) -> None:
        pygame.init()
        pygame.mixer.init(44100, -16, 2, 1024)

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
        self.clock = pygame.time.Clock()
        self.high_scores = HighScores()
        self.sound_effects = SoundEffects()

        # Game state
        self.score = 0
        self.last_extra_life_score = 0  # Track when we last gave an extra life
        self.level = 1
        self.lives = 3
        self.game_over = False
        self.paused = False
        self.num_guns = STARTING_GUNS  # Number of guns to start with (1-4)
        self.entering_name = False
        self.current_name = ""
        self.respawn_timer = 0
        self.new_life_timer = 0  # Timer for new life animation

        # Game objects
        self.ship = None
        self.asteroids: List[Asteroid] = []
        self.bullets: List[Bullet] = []
        self.particles: List[Particle] = []

        # Initialize game
        self.reset_ship()
        self.start_new_level(self.level)
        self.show_level_text = False
        self.level_text_timer = 0  # Add timer for level text

    def reset_ship(self) -> None:
        """Create a new ship in the center of the screen"""
        self.ship = Ship(
            Vector2(self.width / 2, self.height / 2),
            scale=(self.scale_x, self.scale_y)
        )
        self.ship.make_invulnerable(240)  # 4 seconds of invulnerability

    def handle_resize(self, new_width: int, new_height: int) -> None:
        self.width = new_width
        self.height = new_height
        self.scale_x = self.width / DEFAULT_WIDTH
        self.scale_y = self.height / DEFAULT_HEIGHT

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
        """Start a new level with the given number"""
        self.level = level_num
        self.show_level_text = True
        self.level_text_timer = 120  # 2 seconds at 60 FPS

        # Clear any remaining objects
        self.bullets.clear()
        self.asteroids.clear()
        self.particles.clear()

        # Calculate number of asteroids with percentage increase per level
        # For level 1: STARTING_ASTEROIDS
        # For level 2: STARTING_ASTEROIDS * (1 + 10%) = STARTING_ASTEROIDS * 1.1
        # For level 3: STARTING_ASTEROIDS * (1 + 20%) = STARTING_ASTEROIDS * 1.2
        # etc.
        increase_factor = 1 + (ASTEROIDS_LEVEL_INCREASE / 100) * (level_num - 1)
        num_asteroids = round(STARTING_ASTEROIDS * increase_factor)

        for _ in range(num_asteroids):
            # Random position along the edge of the screen
            if random.random() < 0.5:
                x = random.choice([0, self.width])
                y = random.random() * self.height
            else:
                x = random.random() * self.width
                y = random.choice([0, self.height])

            # Create large asteroid
            self.asteroids.append(
                Asteroid(Vector2(x, y), ASTEROID_SIZES['LARGE'], (self.scale_x, self.scale_y))
            )

    def handle_collisions(self) -> None:
        """Handle all game collisions"""
        if self.game_over:
            return

        # Check ship collision with asteroids
        if self.ship and not self.ship.invulnerable:
            for asteroid in self.asteroids[:]:  # Use slice to allow removal during iteration
                if asteroid.check_collision_with_ship(self.ship):
                    self.lives -= 1

                    # Final explosion for game over
                    if self.lives <= 0:
                        # Create a massive explosion
                        self.particles.extend(
                            create_explosion_particles(
                                self.ship.pos,
                                num_particles=50,  # More particles
                                scale=(self.scale_x, self.scale_y),
                                size_range=(2, 6),  # Bigger particles
                                speed_range=(3, 8),  # Faster particles
                                colors=[  # More dramatic colors
                                    (255, 200, 50, 255),  # Bright yellow
                                    (255, 150, 0, 255),   # Orange
                                    (255, 50, 0, 255),    # Red
                                    (255, 0, 0, 255)      # Deep red
                                ]
                            )
                        )
                        # Play the long explosion sound
                        self.sound_effects.play('final_explosion')
                        self.handle_game_over()
                        return

                    # Normal ship destruction
                    ship_pos = self.ship.pos  # Store position before removing ship
                    self.ship = None  # Remove the ship
                    self.respawn_timer = 120  # 2 seconds at 60 FPS
                    self.sound_effects.stop_thrust()  # Stop thrust sound
                    self.sound_effects.play('explosion_large')
                    self.particles.extend(
                        create_explosion_particles(
                            ship_pos,  # Use stored position
                            30,
                            (self.scale_x, self.scale_y)
                        )
                    )
                    break

        # Check bullet collisions with asteroids
        for bullet in self.bullets[:]:  # Use slice to allow removal during iteration
            for asteroid in self.asteroids[:]:  # Use slice to allow removal during iteration
                if asteroid.point_in_asteroid(bullet.pos):
                    # Remove bullet
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)

                    # Split or remove asteroid
                    self.handle_asteroid_destruction(asteroid)
                    break

    def handle_asteroid_destruction(self, asteroid: Asteroid) -> None:
        """Handle the destruction of an asteroid including splitting and scoring"""
        if asteroid in self.asteroids:
            self.asteroids.remove(asteroid)

            # Add explosion particles
            self.particles.extend(
                create_explosion_particles(asteroid.pos, 20, (self.scale_x, self.scale_y))
            )

            # Split asteroid if large enough
            if asteroid.size >= ASTEROID_SIZES['LARGE']:  # Large asteroid
                self.sound_effects.play('explosion_large')
                # Split into two medium asteroids
                for _ in range(2):
                    self.asteroids.append(
                        Asteroid(Vector2(asteroid.pos), ASTEROID_SIZES['MEDIUM'], (self.scale_x, self.scale_y))
                    )
                points = 100
            elif asteroid.size >= ASTEROID_SIZES['MEDIUM']:  # Medium asteroid
                self.sound_effects.play('explosion_medium')
                # Split into two small asteroids
                for _ in range(2):
                    self.asteroids.append(
                        Asteroid(Vector2(asteroid.pos), ASTEROID_SIZES['SMALL'], (self.scale_x, self.scale_y))
                    )
                points = 150
            else:  # Small asteroid
                self.sound_effects.play('explosion_small')
                points = 200

            # Add points
            self.score += points

    def handle_game_over(self) -> None:
        """Handle the game over state"""
        if not self.game_over:
            self.game_over = True
            self.ship = None  # Remove the ship
            self.sound_effects.stop_thrust()  # Stop any ongoing thrust sound

            # Check for high score only if score > 0
            if self.score > 0 and self.high_scores.is_high_score(self.score):
                self.entering_name = True
                self.current_name = ""
            else:
                self.entering_name = False

    def fire_bullet(self) -> None:
        """Fire bullet(s) from the ship's nose"""
        if not self.ship:
            return

        nose_pos = self.ship.get_nose_position()
        base_angle = self.ship.angle

        # Calculate angles based on number of streams
        if self.num_guns == 1:
            angles = [0]  # Single bullet straight ahead
        elif self.num_guns == 2:
            angles = [-5, 5]  # Slightly spread
        elif self.num_guns == 3:
            angles = [-10, 0, 10]  # Wider spread
        else:  # 4 streams
            angles = [-15, -5, 5, 15]  # Widest spread

        # Create bullets at calculated angles
        for angle_offset in angles:
            # Calculate bullet direction based on ship's angle plus offset
            bullet_angle = base_angle + angle_offset
            rad = math.radians(bullet_angle)

            # Direction is opposite to where the ship is pointing
            direction = Vector2(math.sin(rad), -math.cos(rad))

            # All bullets start at the nose position
            self.bullets.append(
                Bullet(nose_pos, direction * 10, (self.scale_x, self.scale_y))
            )
            self.sound_effects.play('fire')

    def reset_game_state(self) -> None:
        """Reset the game to its initial state"""
        self.score = 0
        self.level = 1
        self.lives = 3
        self.game_over = False
        self.paused = False
        self.num_guns = STARTING_GUNS  # Number of guns to start with (1-4)
        self.entering_name = False
        self.current_name = ""
        self.respawn_timer = 0
        self.new_life_timer = 0

        # Clear all game objects
        self.ship = None
        self.asteroids = []
        self.bullets = []
        self.particles = []

        # Stop all sounds
        self.sound_effects.stop_thrust()

        # Initialize game
        self.reset_ship()
        self.start_new_level(self.level)

    def run(self) -> None:
        running = True
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE and not self.game_over and not self.paused:
                        self.fire_bullet()
                    elif event.key == pygame.K_r and self.game_over and not self.entering_name:
                        self.reset_game_state()
                    elif event.key == pygame.K_q:
                        if self.game_over and not self.entering_name:
                            # Quit the game if we're at game over screen
                            running = False
                        elif not self.game_over:
                            # Force game over if we're playing
                            self.handle_game_over()

                    elif event.key == pygame.K_p and not self.game_over:
                        self.paused = not self.paused
                        if self.paused:
                            self.sound_effects.stop_thrust()  # Stop thrust sound when pausing
                            pygame.mixer.pause()  # Pause all sound channels
                        else:
                            pygame.mixer.unpause()  # Unpause all sound channels
                    elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4] and not self.game_over:
                        self.num_guns = int(event.unicode)
                    elif self.entering_name:
                        if event.key == pygame.K_RETURN and self.current_name:
                            self.high_scores.add_score(self.current_name, self.score, self.level)
                            self.entering_name = False
                        elif event.key == pygame.K_BACKSPACE:
                            self.current_name = self.current_name[:-1]
                        elif len(self.current_name) < 10 and event.unicode.isalnum():
                            self.current_name += event.unicode
                elif event.type == pygame.VIDEORESIZE:
                    self.handle_resize(event.w, event.h)

            # Get keyboard state
            keys = pygame.key.get_pressed()

            # Handle respawn timer
            if not self.game_over and not self.paused and self.ship is None and self.lives > 0:
                if self.respawn_timer > 0:
                    self.respawn_timer -= 1
                else:
                    self.reset_ship()

            # Update game objects if not game over or paused
            if not self.game_over and not self.paused:
                # Handle ship movement
                if self.ship:
                    thrust = keys[pygame.K_UP] or keys[pygame.K_w]
                    rotate_left = keys[pygame.K_LEFT] or keys[pygame.K_a]
                    rotate_right = keys[pygame.K_RIGHT] or keys[pygame.K_d]

                    self.ship.update(thrust, rotate_right - rotate_left, self.width, self.height)

                    # Handle thrust sound and particles
                    if thrust:
                        self.sound_effects.start_thrust()
                        self.particles.extend(create_thruster_particles(self.ship.get_rear_position(),
                                           self.ship.angle, (self.scale_x, self.scale_y)))
                    else:
                        self.sound_effects.stop_thrust()

                # Update bullets
                for bullet in self.bullets[:]:
                    bullet.update(self.width, self.height)
                    bullet.lifetime -= 1
                    if bullet.lifetime <= 0:
                        self.bullets.remove(bullet)

                # Update asteroids
                for asteroid in self.asteroids:
                    asteroid.update(self.width, self.height)

                # Update particles
                for particle in self.particles[:]:
                    particle.update()
                    if particle.life <= 0:
                        self.particles.remove(particle)

                # Handle collisions
                self.handle_collisions()

                # Update sound effects beat
                self.sound_effects.update_beat(len(self.asteroids))

            # Update new life animation timer
            if self.new_life_timer > 0:
                self.new_life_timer -= 1

            # Update level text timer
            if self.level_text_timer > 0:
                self.level_text_timer -= 1
                if self.level_text_timer == 0:
                    self.show_level_text = False

            # Draw everything
            self.screen.fill((0, 0, 0))  # Clear screen

            # Draw game objects
            if self.ship:
                self.ship.draw(self.screen)
            for asteroid in self.asteroids:
                asteroid.draw(self.screen)
            for bullet in self.bullets:
                bullet.draw(self.screen)
            for particle in self.particles:
                particle.draw(self.screen)

            # Draw level announcement in center if active
            if self.show_level_text:
                font = pygame.font.SysFont('Arial', int(48 * self.scale_x))
                level_text = font.render(f"Level {self.level}", True, (255, 255, 255))
                level_rect = level_text.get_rect(center=(self.width / 2, self.height / 2))
                self.screen.blit(level_text, level_rect)

            # Draw PAUSED text if game is paused
            if self.paused:
                font = pygame.font.SysFont('Arial', int(48 * self.scale_x))
                pause_text = font.render("PAUSED", True, (255, 255, 0))
                pause_rect = pause_text.get_rect(center=(self.width / 2, self.height / 2))
                self.screen.blit(pause_text, pause_rect)

            # Draw UI
            draw_ui(self.screen, self.score, self.lives, self.level,
                   (self.scale_x, self.scale_y), self.game_over,
                   self.show_level_text, self.entering_name,
                   self.current_name, self.high_scores,
                   self.new_life_timer)

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Game()
    game.run()
