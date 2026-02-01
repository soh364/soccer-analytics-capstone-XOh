from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.process import run_pipeline
from src.utils.config import DATA_DIR, PROCESSED_DIR


def main(in_dir=DATA_DIR, out_dir=PROCESSED_DIR):
    """
    Run the data processing pipeline and print a short summary
    """
    print("Running data pipeline...")
    res = run_pipeline(in_dir, out_dir)

    print(
        f"Validation: {'PASSED' if res.get('validation_passed') else 'FAILED'} | "
        f"matches={res.get('n_matches', 0):,} events={res.get('n_events', 0):,} "
        f"lineups={res.get('n_lineups', 0):,}"
    )


if __name__ == "__main__":
    main()