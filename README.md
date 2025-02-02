## Todos

- fix bullet angles
- error after submit hs name:
uv run game.py
pygame 2.6.1 (SDL 2.28.4, Python 3.13.1)
Hello from the pygame community. https://www.pygame.org/contribute.html
2025-02-02 13:37:29.521 python3[91787:11970439] +[IMKClient subclass]: chose IMKClient_Modern
2025-02-02 13:37:29.521 python3[91787:11970439] +[IMKInputSession subclass]: chose IMKInputSession_Modern
Traceback (most recent call last):
  File "/Users/patiman/git/pygame/asteroids/game.py", line 1164, in <module>
    game.run()
    ~~~~~~~~^^
  File "/Users/patiman/git/pygame/asteroids/game.py", line 1150, in run
    draw_ui(self.screen, self.score, self.lives, self.level,
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           (self.scale_x, self.scale_y), self.game_over,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           self.show_level_text, self.entering_name,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           self.current_name, self.high_scores,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           self.new_life_timer)
           ^^^^^^^^^^^^^^^^^^^^
  File "/Users/patiman/git/pygame/asteroids/game.py", line 715, in draw_ui
    surface.blit(restart_text, restart_text)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: invalid destination position for blit


- [ ] Make asteroids and ships more 3d
- [ ] Add multiplayer challenge where:
    - Player 1 plays on the left side
    - Player 2 plays on the right side
    - Both players have lives and scores
    - Players battle it out for the top score
- [ ] coop mode: players work together to destroy level, no friendly fire or collisions.

