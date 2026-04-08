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

        print(f"Saving merged file to {args.output}")
        save(merged, args.output)

        print("Done ✅")

if __name__ == "__main__":
        main()

