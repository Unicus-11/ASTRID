import subprocess
import sys


SCENARIOS = [
    "baseline",
    "realistic_gps",
    "east_west_heavy",
    "north_south_heavy",
    "high_demand",
    "sensor_degradation",
    "heavy_all_sides"
]


def main():

    for scenario in SCENARIOS:

        print()
        print("=" * 70)
        print(f"RUNNING SCENARIO: {scenario}")
        print("=" * 70)

        subprocess.run(
            [
                sys.executable,
                "-m",
                "sensing.state_extractor",
                scenario
            ],
            check=True
        )


if __name__ == "__main__":
    main()