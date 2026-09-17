import pandas as pd
from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent

raw_dir = project_root / "data" / "raw"
processed_dir = project_root / "data" / "processed"

def preprocess_data():
    """
    Preprocessing data: takes raw dataset from data/raw,
    processes and saves into data/processed
    """

    # Create the folder `processed`, if it does not exist
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Search for CSV-files in `raw`
    raw_files = list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.parquet"))

    if not raw_files:
        print(f"[ERROR] There is no (csv/parquet) files in {raw_dir}")
        sys.exit(1)

    print(f"[INFO] Found files: {len(raw_files)}")

    for raw_file in raw_files:
        print(f"[INFO] Processing: {raw_file.name}")

        # Read the data
        if raw_file.suffix == ".csv":
            df = pd.read_csv(raw_file)
        elif raw_file.suffix == ".parquet":
            df = pd.read_parquet(raw_file)
        else:
            print(f"[WARNING] Unsupported format: {raw_file.suffix}")
            continue

        print(f"[INFO] Original size: {df.shape}")

        # Preprocessing logic

        # 1. Drop the duplicates
        df = df.drop_duplicates()

        # 2. Drop the NaN
        df = df.dropna()

        print(f"[INFO] Size after processing: {df.shape}")

        # Save processed dataset
        processed_file = processed_dir / raw_file.name
        df.to_csv(processed_file, index=False)
        print(f"[SUCCESS] Сохранено: {processed_file}")


if __name__ == "__main__":
    preprocess_data()