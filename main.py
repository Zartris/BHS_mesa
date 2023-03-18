from pathlib import Path

from ABM import server
import argparse
import yaml
import collections.abc

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
    with open(str(Path('configs', config_name)), 'r')as file:
        config_updates = yaml.safe_load(file)
    config = update(config, config_updates)
    server.run(config)
