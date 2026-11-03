#!/usr/bin/env python3
"""``ros2 run cat_arena generate_arena.py`` - thin CLI wrapper around
:func:`cat_arena.generate.main`."""

from cat_arena.generate import main

if __name__ == "__main__":
    raise SystemExit(main())
