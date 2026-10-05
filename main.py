"""Entry point:  python main.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cli.app import CLI  # noqa: E402


def main() -> int:
    try:
        CLI().run()
    except ValueError as exc:        # e.g. invalid settings file
        print(f"Configuration error: {exc}")
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
