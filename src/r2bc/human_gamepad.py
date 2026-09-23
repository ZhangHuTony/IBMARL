"""Small pygame Xbox adapter used by the human R2BC collector.

The mapping uses the raw SDL layout measured by ``calibrate_gamepad`` on the
collection controller: left-stick horizontal is axis 0, vertical is axis 1,
and A is button 0. VMAS uses the same X/Y action order, so the axes must not be
swapped; only SDL's downward-positive vertical axis is negated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GamepadState:
    action: np.ndarray
    start: bool = False
    reject: bool = False
    quit: bool = False


class XboxGamepad:
    """Poll one Xbox-compatible controller without owning the render window."""

    A = 0
    B = 1
    START = 7

    def __init__(
        self,
        index: int = 0,
        speed: float = 1.0,
        precision_speed: float = 0.25,
        deadzone: float = 0.1,
        swap_axes: bool = False,
        invert_x: bool = False,
        invert_y: bool = False,
    ):
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "Human R2BC needs pygame. Install the pinned requirements with "
                "`pip install -r requirements.txt`."
            ) from exc

        self.pygame = pygame
        pygame.init()
        pygame.joystick.init()
        count = pygame.joystick.get_count()
        if not 0 <= index < count:
            raise RuntimeError(
                f"Xbox controller index {index} is unavailable; detected {count} joystick(s)."
            )
        self.joystick = pygame.joystick.Joystick(index)
        self.joystick.init()
        self.speed = speed
        self.precision_speed = precision_speed
        self.deadzone = deadzone
        self.swap_axes = swap_axes
        self.invert_x = invert_x
        self.invert_y = invert_y
        self._previous_buttons: set[int] = set()
        print(f"Xbox controller: {self.joystick.get_name()}")

    def map_stick(self, raw_x: float, raw_y: float) -> np.ndarray:
        """Map pygame axes into VMAS action coordinates.

        Pygame reports physical up as negative Y, while VMAS actions are
        Cartesian ``[x, y]``. Thus physical right maps to ``[+1, 0]`` and up
        maps to ``[0, +1]``.
        """
        raw_x = 0.0 if abs(raw_x) < self.deadzone else raw_x
        raw_y = 0.0 if abs(raw_y) < self.deadzone else -raw_y

        action = np.asarray([raw_x, raw_y], dtype=np.float32)
        if self.invert_x:
            action[0] *= -1
        if self.invert_y:
            action[1] *= -1
        if self.swap_axes:
            action = action[::-1].copy()
        return action

    def poll(self) -> GamepadState:
        self.pygame.event.pump()
        raw_x = self.joystick.get_axis(0)
        raw_y = self.joystick.get_axis(1)
        action = self.map_stick(raw_x, raw_y)

        trigger = self.joystick.get_axis(5) if self.joystick.get_numaxes() > 5 else -1.0
        action *= self.precision_speed if trigger > 0.1 else self.speed
        action = np.clip(action, -1.0, 1.0)

        pressed = {
            i for i in range(self.joystick.get_numbuttons())
            if self.joystick.get_button(i)
        }
        edges = pressed - self._previous_buttons
        self._previous_buttons = pressed
        return GamepadState(
            action=action,
            start=self.A in edges,
            reject=self.B in edges,
            quit=self.START in edges,
        )

    def close(self) -> None:
        self.joystick.quit()
        self.pygame.joystick.quit()
        self.pygame.quit()
