import argparse
import os
import pickle
import re

import numpy as np
import xarray as xa


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'experiment_dir',
        help="Path to directory where the results of experiment defined in "
             "one config lay."
    )
    parser.add_argument(
        'tmpl',
        help="A regular expression matching files with experiments results. "
             "It has to be a path given relative to 'experiment_dir'. It also "
             "has to define groups extracting values of hyper parameters."
    )
    parser.add_argument(
        '--types',
        '-t',
        help="Variable types of hyper parameters extracted from path.",
        nargs='+'
    )
    parser.add_argument(
        "--dims",
        "-d",
        help="Names of dimensions in constructed DataArray. It has to be equal"
             "to number of hyper parameters extracted with 'tmpl' + 1 where "
             "the last dim is values of collected metrics.",
        nargs='+',
    )
    parser.add_argument(
        "--save_path",
        "-s",
        help="Path to file where DataArray is saved. Path is given relative "
             "to 'experiment_dir'.",
        default='results.netcdf'
    )
    return parser.parse_args()


def find_and_parse_paths_to_files(tmpl, types):
    types = [eval(t) for t in types]
    paths, parsed_values = [], []
    for path, dirs, files in os.walk('.'):
        for f in files:
            p = os.path.join(path, f)
            # print(p)
            parsed = re.match(tmpl, p)
            if parsed is not None:
                paths.append(p)
                gs = parsed.groups()
                gs = list(filter(lambda x: x is not None, gs))
                if len(gs) != len(types):
                    raise ValueError(
                        "Number of types used for conversion does not match "
                        "number of extracted groups."
                    )
                gs = [t(g) for t, g in zip(types, gs)]
                parsed_values.append(gs)
    return paths, parsed_values


def get_indices_by_dims(parsed_by_paths):
    all_extracted_values = zip(*parsed_by_paths)
    indices_by_dims = []
    for one_dim_values in all_extracted_values:
        sorted_unique = np.sort(np.unique(one_dim_values))
        one_dim_unique = {}
        for i, v in enumerate(sorted_unique):
            one_dim_unique[v] = i
        indices_by_dims.append(one_dim_unique)
    return indices_by_dims


def get_indices_for_launch(parsed, indices_by_dims):
    indices = []
    for i, p in enumerate(parsed):
        indices.append(indices_by_dims[i][p])
    return tuple(indices)


def extract_data_from_txt(path):
    steps, values = [], []
    with open(path) as f:
        for line in f:
            step, value = line.split()
            steps.append(int(step))
            values.append(float(value))
    return steps, values


def extract_data_from_pickle(path):
    values = []
    with open(path, 'rb') as f:
        while True:
            try:
                values.append(pickle.load(f))
            except EOFError:
                break
    return np.array(values)


def extract_data(paths, parsed_by_paths):
    indices_by_dims = get_indices_by_dims(parsed_by_paths)
    shape = [len(i) for i in indices_by_dims]
    data = np.zeros(shape, dtype=object)
    longest_steps = []
    for path, parsed in zip(paths, parsed_by_paths):
        _,  ext = os.path.splitext(path)
        if ext == '.txt':
            steps, values = extract_data_from_txt(path)
            if len(steps) > len(longest_steps):
                longest_steps = steps
        elif ext == '.pickle':
            values = extract_data_from_pickle(path)
        else:
            raise ValueError(
                "Only pickle and txt files with experiment "
                "results are allowed. Found {}".format(repr(path)))
        values = np.array(values)
        indices = get_indices_for_launch(parsed, indices_by_dims)
        data[tuple(indices)] = values
    return data, longest_steps


def complete_array_of_arrays(data):
    max_len = np.max(np.vectorize(len)(data))
    shape = data.shape
    data = data.reshape([-1])
    data = list(
        map(lambda x: np.r_[x, np.full([max_len-x.size], np.nan)], data))
    data = np.stack(data)
    data = data.reshape(shape + (max_len,))
    return data


def main():
    args = parse_args()
    old_dir = os.getcwd()
    os.chdir(args.experiment_dir)
    paths, parsed = find_and_parse_paths_to_files(args.tmpl, args.types)
    paths, parsed = zip(*sorted(zip(paths, parsed), key=lambda x: x[1]))
    data, steps = extract_data(paths, parsed)
    data = complete_array_of_arrays(data)
    coords = [np.unique(coord_values) for coord_values in zip(*parsed)]
    coords.append(steps)
    data_array = xa.DataArray(
        data,
        dims=args.dims,
        coords=list(zip(args.dims, coords))
    )
    data_array.to_netcdf(args.save_path)
    os.chdir(old_dir)


if __name__ == '__main__':
    main()