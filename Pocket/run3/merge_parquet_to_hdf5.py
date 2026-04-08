#!/usr/bin/env python3

import argparse
import glob
import pandas as pd

def merge_parquet_to_hdf5(
    parquet_files,
    output_hdf5,
    key="data",
    chunksize=None,
    columns=None,
):
    first = True

    for pq in parquet_files:
        print(f"Processing {pq}")

        if chunksize is None:
            df = pd.read_parquet(pq, columns=columns)
            df.to_hdf(
                output_hdf5,
                key=key,
                mode="w" if first else "a",
                format="table",
                append=not first,
                data_columns=True,
            )
            first = False
        else:
            # Chunked reading using pyarrow
            import pyarrow.parquet as pqfile

            pf = pqfile.ParquetFile(pq)
            for i, batch in enumerate(pf.iter_batches(batch_size=chunksize, columns=columns)):
                df = batch.to_pandas()
                df.to_hdf(
                    output_hdf5,
                    key=key,
                    mode="w" if first else "a",
                    format="table",
                    append=not first,
                    data_columns=True,
                )
                first = False

    print(f"Merged {len(parquet_files)} parquet files into {output_hdf5}")

def main():
    parser = argparse.ArgumentParser(description="Merge Parquet files into an HDF5 file")
    parser.add_argument("parquet", nargs="+", help="Input parquet files or glob patterns")
    parser.add_argument("-o", "--output", required=True, help="Output HDF5 file")
    parser.add_argument("--key", default="data", help="HDF5 key (default: data)")
    parser.add_argument("--chunksize", type=int, default=None, help="Chunk size for large files")
    parser.add_argument("--columns", nargs="+", default=None, help="Columns to keep")

    args = parser.parse_args()

    parquet_files = []
    for p in args.parquet:
        parquet_files.extend(glob.glob(p))

    if not parquet_files:
        raise RuntimeError("No parquet files found")

    merge_parquet_to_hdf5(
        parquet_files,
        args.output,
        key=args.key,
        chunksize=args.chunksize,
        columns=args.columns,
    )

if __name__ == "__main__":
    main()
