# Asteroids Game

This repository contains a Pygame implementation of the classic Asteroids game.

## Improvements over the Original

This version includes the following improvements:

*   **Modernized Graphics:** Enhanced visuals using Pygame.
*   **Synthesized Sound Effects:** The game uses synthesized sound effects created with `pygame.mixer`, offering a more modern and customizable audio experience.
*   **Dynamic Level Scaling:** The number of asteroids increases with each level, providing a progressive challenge. The rate of increase is configurable via the `ASTEROIDS_LEVEL_INCREASE` setting in `game.yaml`.
*   **Configurable Asteroids:** The size of asteroids is configurable using the `ASTEROID_SIZES` setting in `game.yaml`, and their speed increases with each level based on the `ASTEROIDS_LEVEL_SPEED_INCREASE` setting in `game.yaml`.
*   **Multiple Guns:** The ship can fire multiple bullets simultaneously, with the number of guns being configurable (1-4) using the number keys. The starting number of guns is set by the `STARTING_GUNS` setting in `game.yaml`.
*   **High Score Tracking:** The game tracks and displays high scores, adding a competitive element.
*   **Invulnerability:** The ship is invulnerable for a short period after respawning, giving the player a chance to recover.
*   **Particle Effects:** The game uses particle effects for explosions and thrusters, enhancing the visual experience.
*   **Window Resizing:** The game window is resizable, and the game adapts to different screen sizes while maintaining the aspect ratio.
*   **More Complex Asteroid Shapes:** The asteroids have more complex and irregular shapes, created using a random point generation algorithm.
*   **Background Beat:** The game includes a background beat that increases in tempo as the number of asteroids decreases, adding to the tension.
*   **Responsive Controls:** Smoother and more accurate controls.

## How to Run

1.  Make sure you have Python installed.
2.  Install the required dependencies using pip:

    ```bash
    pip install pygame pyyaml
    ```

    Or, install the required dependencies using uv:

    ```bash
    uv pip install .
    ```
3.  Run the game:

    ```bash
    python game.py
    ```

## Configuration

The game's settings can be configured by modifying the `game.yaml` file.
