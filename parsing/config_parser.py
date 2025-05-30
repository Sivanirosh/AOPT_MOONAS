import yaml

class ConfigParser:
    def __init__(self, config_file_path):
        with open(config_file_path, "r") as f:
            self.cfg = yaml.safe_load(f)

    def get(self, *keys):
        node = self.cfg
        for k in keys:
            node = node[k]
        return node
