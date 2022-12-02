import json
import argparse
from trainer import train
import yaml

def load(settings_path):
    with open(settings_path, encoding='gb18030', errors='ignore') as data_file:
        if 'json' in settings_path:
            param = json.load(data_file)
        elif 'yml' in settings_path:
            param = yaml.load(data_file,Loader=yaml.FullLoader)
    return param

def main():
    args = setup_parser().parse_args()
    param = load(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='incremental learning algorithms.')
    parser.add_argument('--config', type=str, default='exps/icarl/cifar100_B0_2tasks.yml',
                        help='Json file of settings.')
    return parser


if __name__ == '__main__':
    main()
