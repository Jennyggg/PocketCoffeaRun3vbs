#!/usr/bin/env python3

import argparse
import glob
import pyarrow as pa
import pyarrow.parquet as pq

def merge_parquet_files(inputs, output, columns=None, compression="zstd"):
    tables = []

    for f in inputs:
        print(f"Reading {f}")
        table = pq.read_table(f, columns=columns)
        tables.append(table)

    print("Concatenating tables...")
    merged = pa.concat_tables(tables, promote=True)

    print(f"Writing {output}")
    pq.write_table(
        merged,
        output,
        compression=compression,
        use_dictionary=True,
        write_statistics=True,
    )

def main():
    parser = argparse.ArgumentParser(description="Merge Parquet files into one Parquet file")
    parser.add_argument("parquet", nargs="+", help="Input parquet files or glob patterns")
    parser.add_argument("-o", "--output", required=True, help="Output parquet file")
    parser.add_argument("--columns", nargs="+", default=None, help="Columns to keep")
    parser.add_argument("--compression", default="zstd", help="Compression codec (default: zstd)")

    args = parser.parse_args()

    files = []
    for p in args.parquet:
        files.extend(glob.glob(p))

    if not files:
        raise RuntimeError("No parquet files found")

    merge_parquet_files(files, args.output, args.columns, args.compression)

if __name__ == "__main__":
    main()
