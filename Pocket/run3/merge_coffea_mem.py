#!/usr/bin/env python3
import argparse
import gc
from coffea.util import load, save
from coffea.processor.accumulator import accumulate

def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient merge of multiple .coffea files"
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
    parser.add_argument(
        "-sr", "--sr",
        required=True,
        help="Name of signal region for columns"
    )
    args = parser.parse_args()

    print(f"Incrementally merging {len(args.inputs)} files...")

    # Load first file
    merged = load(args.inputs[0])
    print(f"Loaded {args.inputs[0]}")

    for key1 in merged['columns'].keys():
        for key2 in merged['columns'][key1].keys():
            key3_pop = []
            for key3 in merged['columns'][key1][key2].keys():
                if key3 != args.sr:
                    key3_pop.append(key3)
            for key3 in key3_pop:
                merged['columns'][key1][key2].pop(key3,None)
    # Merge remaining files one by one
    for i, fname in enumerate(args.inputs[1:], start=2):
        print(f"[{i}/{len(args.inputs)}] Merging {fname}")
        acc = load(fname)
        for key1 in acc['columns'].keys():
            for key2 in acc['columns'][key1].keys():
                key3_pop = []
                for key3 in acc['columns'][key1][key2].keys():
                    if key3 != args.sr:
                        key3_pop.append(key3)
                for key3 in key3_pop:
                    acc['columns'][key1][key2].pop(key3,None)
        merged = accumulate([merged, acc])

        # Free memory ASAP
        del acc
        gc.collect()

    print(f"Saving merged file to {args.output}")
    save(merged, args.output)

    print("Done ✅")

if __name__ == "__main__":
    main()
