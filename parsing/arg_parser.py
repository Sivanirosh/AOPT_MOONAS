# parsing/arg_parser.py

import argparse

class ArgParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            '--device-id', type=int, default=0,
            help="CUDA device ID (default: 0)"
        )
        self.parser.add_argument(
            '--config-file-path', type=str, required=True,
            help="Path to YAML config file"
        )
        self.parser.add_argument(
            '--experiment-name', type=str, required=True,
            help="Unique name for this experiment"
        )
        self.parser.add_argument(
            '--checkpoint-dir', type=str, default=None,
            help="Directory to save/load checkpoints (default: ./checkpoints/<experiment-name>)"
        )

        self.parser.add_argument(
            '--num-epochs', type=int, default=10,
            help="Number of training epochs"
        )
        self.parser.add_argument(
            '--num-workers', type=int, default=4,
            help="DataLoader num_workers"
        )
        self.parser.add_argument(
            '--resume', action='store_true',
            help="Resume training from latest checkpoint"
        )
        self.parser.add_argument(
            '--no-log', action='store_true',
            help="Disable logging"
        )

    def parse_args(self):
        args = self.parser.parse_args()
        # if user didn't pass --checkpoint-dir, default based on experiment-name
        if args.checkpoint_dir is None:
            args.checkpoint_dir = f"./checkpoints/{args.experiment_name}"
        return args
