import argparse

import numpy as np
import xarray as xa


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'in_file',
        help="Path to file with experiments results in netcdf format."
    )
    parser.add_argument(
        "--out_file",
        help="Path to file where results will be stored. By default it is "
             "equal to `in_file`."
    )
    args = parser.parse_args()
    if args.out_file is None:
        args.out_file = args.in_file
    return args


def sqrt_correlation(in_file, out_file):
    da = xa.open_dataarray(in_file)
    corr_metrics = [
        np.where(da.coords['metric'] == c)[0][0] for c
        in da.coords['metric'] if "corr" in str(c.data)]
    print(in_file, corr_metrics)
    indexer = dict(metric=corr_metrics)
    da[indexer] = da[indexer] ** 0.5
    da.to_netcdf(out_file)


def main():
    args = parse_args()
    sqrt_correlation(args.in_file, args.out_file)


if __name__ == '__main__':
    main()
