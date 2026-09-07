import sys
from pathlib import Path

CONTROLLER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONTROLLER_DIR.parent
SUMO_DIR = PROJECT_ROOT / "sumo"

sys.path.insert(0, str(CONTROLLER_DIR))

from sumo_interface import SumoInterface, LoopConfig


def keep_policy(state):
    """Test policy: always KEEP the current stage."""
    return "KEEP"


def main():
    import sumo_interface

    original_policy = sumo_interface.placeholder_policy
    sumo_interface.placeholder_policy = keep_policy

    config_path = SUMO_DIR / "Squire_Junction_Multiple_Lanes" / "sq.sumo.cfg"

    interface = SumoInterface(
        LoopConfig(
            sumo_binary="sumo",
            config_path=str(config_path),
            max_steps=70,
            print_every_s=1.0,
        )
    )

    try:
        interface.start()
        interface.run()
    finally:
        interface.close()
        sumo_interface.placeholder_policy = original_policy


if __name__ == "__main__":
    main()