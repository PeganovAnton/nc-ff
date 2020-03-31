import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xa


COLORS = [
    'r', 'g', 'b', 'k', 'c', 'magenta', 'brown',
    'darkviolet', 'pink', 'yellow', 'gray', 'orange', 'olive',
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_path',
        help="Path to file with DataArray."
    )
    parser.add_argument(
        '--save_dir',
        '-s',
        help="Path to directory with plots."
    )
    parser.add_argument(
        '--plot_dim',
        '-p',
        help="The name of the dimension which value is fixed for a plot."
    )
    parser.add_argument(
        '--color_dim',
        '-c',
        help="The name of the dimension which is varied on 1 plot with color "
             "changing."
    )
    parser.add_argument(
        '--beam_dim',
        '-b',
        help="The name of the dimension which is varied in 1 beam."
    )
    parser.add_argument(
        '--config',
        help="Path to JSON config with plotting parameters. If parameters "
             "were passed to script directly they have higher priority "
             "compared with config."
    )
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    config['data_path'] = args.data_path
    if args.save_dir is not None:
        config['save_dir'] = args.save_dir
    if args.plot_dim is not None:
        config['plot_dim'] = args.plot_dim
    if args.color_dim is not None:
        config['color_dim'] = args.color_dim
    if args.beam_dim is not None:
        config['beam_dim'] = args.beam_dim
    return config


def get_color(idx):
    return COLORS[idx]


def plot_beam(da, config, color, color_coord):
    mean = da.reduce(np.mean, config['beam_dim'])
    plt.plot(
        da.coords['step'],
        mean,
        lw=1.5,
        color=color,
        label=config['color_dim'] + ': ' + str(color_coord.data)
    )
    for beam_coord in da.coords[config['beam_dim']]:
        plt.plot(
            da.coords['step'],
            da.sel(**{config['beam_dim']: beam_coord}),
            lw=0.3,
            color=color,
        )


def plot_fill_band(da, config, color, color_coord):
    mean = da.reduce(np.mean, config['beam_dim'])
    std = da.reduce(np.std, config['beam_dim'], ddof=1)
    plt.plot(
        da.coords['step'],
        mean,
        lw=1.5,
        color=color,
        label=config['color_dim'] + ': ' + str(color_coord.data)
    )
    plt.fill_between(
        da.coords['step'],
        mean-std,
        mean+std,
        color=color,
        alpha=0.2
    )


def draw_beams_plot(da, save_file, config, plot_coord):
    fig, ax = plt.subplots()
    for i, color_coord in enumerate(da.coords[config['color_dim']]):
        color = get_color(i)
        if config['plot_type'] == 'beam':
            plot_beam(
                da.sel(
                    **{config['color_dim']: color_coord}),
                config,
                color,
                color_coord
            )
        elif config['plot_type'] == 'fill':
            plot_fill_band(
                da.sel(
                    **{config['color_dim']: color_coord}),
                config,
                color,
                color_coord
            )
        else:
            raise ValueError("Unsupported plot type {}".format(
                config['plot_type']))
    ax.grid()
    ax.set_xlabel('step')
    ax.set_xscale('log')
    ax.set_ylabel(config['ylabels'][str(plot_coord.data)])
    ax.legend()
    plt.tight_layout()
    dir_, file = os.path.split(save_file)
    os.makedirs(dir_, exist_ok=True)
    plt.savefig(save_file, dpi=900)


def draw_beams_plots(da, config):
    for plot_coord in da.coords[config['plot_dim']]:
        print(plot_coord.data)
        draw_beams_plot(
            da.sel(**{config['plot_dim']: plot_coord}),
            os.path.join(config['save_dir'], str(plot_coord.data) + '.png'),
            config,
            plot_coord
        )


def main():
    config = parse_args()
    da = xa.load_dataarray(config['data_path'])
    draw_beams_plots(
        da,
        config
    )


if __name__ == '__main__':
    main()