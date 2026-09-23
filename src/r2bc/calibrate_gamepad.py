"""Report the raw SDL mapping for the gesture used by human R2BC.

Run this before changing controller transforms. It deliberately makes no Xbox
axis/button assumptions: the requested stick gesture and button press identify
the relevant raw controls on the operator's actual SDL/controller mapping.
"""

from __future__ import annotations

import argparse
import time

import pygame


def _axes(joystick) -> list[float]:
    pygame.event.pump()
    return [joystick.get_axis(i) for i in range(joystick.get_numaxes())]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Inspect raw Xbox/SDL controls.")
    parser.add_argument("--controller", type=int, default=0)
    args = parser.parse_args(argv)

    pygame.init()
    pygame.joystick.init()
    count = pygame.joystick.get_count()
    if not 0 <= args.controller < count:
        raise SystemExit(
            f"Controller {args.controller} unavailable; SDL detected {count} joystick(s)."
        )

    joystick = pygame.joystick.Joystick(args.controller)
    joystick.init()
    print(f"Detected: {joystick.get_name()}", flush=True)
    print(
        f"GUID: {joystick.get_guid()} | axes={joystick.get_numaxes()} | "
        f"buttons={joystick.get_numbuttons()} | hats={joystick.get_numhats()}",
        flush=True,
    )
    print("Release the sticks and buttons; measuring neutral for one second...", flush=True)

    neutral_samples = []
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        neutral_samples.append(_axes(joystick))
        time.sleep(0.01)
    neutral = [
        sum(sample[i] for sample in neutral_samples) / len(neutral_samples)
        for i in range(joystick.get_numaxes())
    ]
    print("Neutral axes: " + ", ".join(f"{i}={v:+.3f}" for i, v in enumerate(neutral)), flush=True)
    print("\nNOW hold the LEFT STICK fully LEFT, then press and hold A.", flush=True)
    print("The script will capture once it sees both a moved axis and a button.", flush=True)

    while True:
        raw = _axes(joystick)
        moved = [
            (i, value, value - neutral[i])
            for i, value in enumerate(raw)
            if abs(value - neutral[i]) > 0.45
        ]
        pressed = [
            i for i in range(joystick.get_numbuttons())
            if joystick.get_button(i)
        ]
        hats = [joystick.get_hat(i) for i in range(joystick.get_numhats())]
        if moved and pressed:
            print("\nCAPTURED", flush=True)
            print("Raw axes: " + ", ".join(f"{i}={v:+.3f}" for i, v in enumerate(raw)), flush=True)
            print(
                "Moved from neutral: "
                + ", ".join(f"axis {i}: raw={v:+.3f}, delta={d:+.3f}" for i, v, d in moved),
                flush=True,
            )
            print(f"Pressed button indices: {pressed}", flush=True)
            print(f"Hat values: {hats}", flush=True)
            break
        time.sleep(0.01)

    joystick.quit()
    pygame.joystick.quit()
    pygame.quit()


if __name__ == "__main__":
    main()
