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
        '--color_dim',
        '-c',
        help="The name of the dimension which is varied on 1 plot with color "
             "changing."
    )
    parser.add_argument(
        '--metric_dim',
        '-b',
        help="The name of the dimension which is varied in 1 beam."
    )
    parser.add_argument(
        '--x_metric',
        help="The name of a metric for x axis, e.g. 'ce_loss'."
    )
    parser.add_argument(
        '--y_metrics',
        help="The list of names of a metrics for y axis, e.g. 'hs0_corr'. "
             "This is also list of plot names.",
        nargs='+',
    )
    parser.add_argument(
        "--averaging_dim",
        help="A dimension used for averaging, e.g. 'launch_number'."
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
    if args.metric_dim is not None:
        config['metric_dim'] = args.metric_dim
    if args.x_metric is not None:
        config['x_metric'] = args.x_metric
    if args.y_metrics is not None:
        config['y_metrics'] = args.y_metrics
    if args.color_dim is not None:
        config['color_dim'] = args.color_dim
    if args.averaging_dim is not None:
        config['averaging_dim'] = args.averaging_dim
    return config


def get_color(idx):
    return COLORS[idx]


def plot_average_bar(da, config, color, color_coord, y_metric):
    mean = da.reduce(np.mean, config['averaging_dim'])
    std = da.reduce(np.std, config['averaging_dim'], ddof=1)
    plt.errorbar(
        mean.sel(**{config['metric_dim']: config['x_metric']}),
        mean.sel(**{config['metric_dim']: y_metric}),
        xerr=std.sel(**{config['metric_dim']: config['x_metric']}),
        yerr=std.sel(**{config['metric_dim']: y_metric}),
        lw=1.5,
        color=color,
        label=config['color_dim'] + ': ' + str(color_coord)
    )


def plot_beam(da, config, color, color_coord, y_metric):
    mean = da.reduce(np.mean, config['averaging_dim'])
    plt.plot(
        mean.sel(**{config['metric_dim']: config['x_metric']}),
        mean.sel(**{config['metric_dim']: y_metric}),
        lw=1.5,
        color=color,
        label=config['color_dim'] + ': ' + str(color_coord)
    )
    for beam_coord in da.coords[config['averaging_dim']]:
        ray_da = da.sel(**{config['averaging_dim']: beam_coord})
        plt.plot(
            ray_da.sel(**{config['metric_dim']: config['x_metric']}),
            ray_da.sel(**{config['metric_dim']: y_metric}),
            lw=0.3,
            color=color
        )


def plot_only_mean(da, config, color, color_coord, y_metric):
    mean = da.reduce(np.mean, config['averaging_dim'])
    plt.plot(
        mean.sel(**{config['metric_dim']: config['x_metric']}),
        mean.sel(**{config['metric_dim']: y_metric}),
        lw=1.5,
        color=color,
        label=config['color_dim'] + ': ' + str(color_coord)
    )


def draw_m_vs_m_plot(da, save_file, config, y_metric):
    fig, ax = plt.subplots()
    if 'color_values' not in config or config['color_values'] is None:
        color_coords = list(
            map(lambda x: x.data, da.coords[config['color_dim']]))
    else:
        color_coords = config['color_values']
    for i, color_coord in enumerate(color_coords):
        color = get_color(i)
        if config['plot_type'] == 'bar':
            plot_average_bar(
                da.sel(**{config['color_dim']: color_coord}),
                config,
                color,
                color_coord,
                y_metric
            )
        elif config['plot_type'] == 'beam':
            plot_beam(
                da.sel(**{config['color_dim']: color_coord}),
                config,
                color,
                color_coord,
                y_metric
            )
        elif config['plot_type'] == 'only_mean':
            plot_only_mean(
                da.sel(**{config['color_dim']: color_coord}),
                config,
                color,
                color_coord,
                y_metric
            )
        else:
            raise ValueError("Unsupported plot type {}".format(
                config['plot_type']))
    ax.grid()
    ax.set_xlabel(config["axis_labels"][config['x_metric']])
    ax.set_ylabel(config['axis_labels'][y_metric])
    ax.set_xscale(config['xscale'])
    if 'xlim' in config:
        ax.set_xlim(*config['xlim'])
    if 'ylim' in config:
        ax.set_ylim(*config['ylim'])
    ax.legend()
    plt.tight_layout()
    dir_, file = os.path.split(save_file)
    os.makedirs(dir_, exist_ok=True)
    plt.savefig(save_file, dpi=900)


def draw_m_vs_m_plots(da, config):
    for y_metric in config['y_metrics']:
        print(y_metric)
        draw_m_vs_m_plot(
            da.sel(**{config['metric_dim']: [config['x_metric'], y_metric]}),
            os.path.join(config['save_dir'], y_metric + '.png'),
            config,
            y_metric
        )


def main():
    config = parse_args()
    da = xa.load_dataarray(config['data_path'])
    draw_m_vs_m_plots(
        da,
        config
    )


if __name__ == '__main__':
    main()