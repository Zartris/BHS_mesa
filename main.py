from pathlib import Path

from ABM import server
import argparse
import yaml
import collections.abc

from ABM.model import AirportModel


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


if __name__ == '__main__':
    # Load user args
    parser = argparse.ArgumentParser()
    parser.add_argument('--cn', type=str, default='config.yaml', help='Name of the config file to use')

    # read args
    args = parser.parse_args()
    config_name = args.cn

    # First we load the default configs:
    with open(str(Path('configs', 'defualt_config.yaml')), 'r') as file:
        config = yaml.safe_load(file)

    # Then we load the updates to the config file
    with open(str(Path('configs', config_name)), 'r') as file:
        config_updates = yaml.safe_load(file)
    config = update(config, config_updates)
    if not config['performance_test']:
        server.run(config)
    else:
        GRID_WIDTH = config['grid_width']
        GRID_HEIGHT = config['grid_height']
        AirportModel(config, GRID_WIDTH, GRID_HEIGHT).run_performance_test(1000)

        # For line profiling use:
        # @profile but don't import anything
        # Then run: kernprof -lv main.py
