#!/usr/bin/env python3
import sys
import argparse
from coffea.util import load, save
from coffea.processor.accumulator import accumulate

def main():
        parser = argparse.ArgumentParser(
                description="Merge multiple .coffea files (from pocket-coffea --process-separately) into one."
        )
        parser.add_argument(
                "inputs",
                nargs="+",
                help="Input .coffea files to merge"
        )
        parser.add_argument(
                "-o", "--output",
                required=True,
                help="Output .coffea filename"
        )

        args = parser.parse_args()

        print(f"Loading {len(args.inputs)} input files...")
        accumulators = [load(fname) for fname in args.inputs]

        print("Merging accumulators...")
        merged = accumulate(accumulators)

        # Fix string metadata duplication from accumulate (e.g., 'True'*3 -> 'True')
        by_dataset = merged.get('datasets_metadata', {}).get('by_dataset', {})
        for dataset_meta in by_dataset.values():
            for key, val in list(dataset_meta.items()):
                if isinstance(val, str) and len(val) > 1:
                    n = len(val)
                    for period in range(1, n // 2 + 1):
                        if n % period == 0 and val[:period] * (n // period) == val:
                            dataset_meta[key] = val[:period]
                            break

        print(f"Saving merged file to {args.output}")
        save(merged, args.output)

        print("Done ✅")

if __name__ == "__main__":
        main()

