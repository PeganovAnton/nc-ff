import argparse
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
    return parser.parse_args()


def get_color(idx):
    return COLORS[idx]


def plot_beam(da, color, beam_dim, color_coord):
    mean = da.reduce(np.mean, beam_dim)
    plt.plot(
        da.coords['values'],
        mean,
        lw=1,
        color=color,
        label=beam_dim + ': ' + str(color_coord)
    )
    for beam_coord in da.coords[beam_dim]:
        plt.plot(
            da.coords['values'],
            da.sel(**{beam_dim: beam_coord}),
            lw=0.2,
            color=color,
        )


def draw_beams_plot(da, save_file, plot_dim, color_dim, beam_dim):
    fig, ax = plt.subplots()
    for i, color_coord in enumerate(da.coords[color_dim]):
        color = get_color(i)
        plot_beam(
            da.sel(
                **{color_dim: color_coord}),
            color,
            beam_dim,
            color_coord.data
        )
    ax.grid()
    ax.set_xlabel('step')
    ax.set_xscale('log')
    ax.set_ylabel(plot_dim)
    ax.legend()
    plt.tight_layout()
    dir_, file = os.path.split(save_file)
    os.makedirs(dir_, exist_ok=True)
    plt.savefig(save_file, dpi=900)


def draw_beams_plots(da, save_dir, plot_dim, color_dim, beam_dim):
    for plot_coord in da.coords[plot_dim]:
        print(plot_coord.data)
        draw_beams_plot(
            da.sel(**{plot_dim: plot_coord}),
            os.path.join(save_dir, str(plot_coord.data) + '.png'),
            plot_dim,
            color_dim,
            beam_dim
        )


def main():
    args = parse_args()
    da = xa.load_dataarray(args.data_path)
    draw_beams_plots(
        da,
        args.save_dir,
        args.plot_dim,
        args.color_dim,
        args.beam_dim
    )


if __name__ == '__main__':
    main()