import os

import torch
import yaml


class AttrDict(dict):
    """A dictionary with attribute-style access. It maps attribute access to
    the real dictionary.
    """

    DEFAULT_CONFIG_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config.yaml"
    )

    def __init__(self, *args, **kwargs):
        dict.__init__(self)

        if len(args) > 1:
            raise TypeError("AttrDict accepts at most one positional argument")

        if len(args) == 1:
            src = args[0]
            if isinstance(src, str):
                src = self._load_yaml(src)
            if not isinstance(src, dict):
                raise TypeError("AttrDict positional argument must be a mapping or config path")
            for k, v in src.items():
                self[k] = self._to_attr_dict(v)

        for k, v in kwargs.items():
            self[k] = self._to_attr_dict(v)

        self._set_runtime_fields()

    @staticmethod
    def _load_yaml(config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise ValueError("Config file must contain a top-level mapping.")

        return data

    @classmethod
    def _to_attr_dict(cls, value):
        if isinstance(value, dict):
            node = cls()
            for k, v in value.items():
                node[k] = cls._to_attr_dict(v)
            return node
        if isinstance(value, list):
            return [cls._to_attr_dict(v) for v in value]
        return value

    def _set_runtime_fields(self):
        system_cfg = self.get("system")
        if not isinstance(system_cfg, dict):
            return
        cuda_enabled = bool(system_cfg.get("cuda", False)) and torch.cuda.is_available()
        system_cfg["device"] = torch.device("cuda") if cuda_enabled else torch.device("cpu")

    def __getstate__(self):
        return self.__dict__.items()

    def __setstate__(self, items):
        for key, val in items:
            self.__dict__[key] = val

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, dict.__repr__(self))

    def __setitem__(self, key, value):
        return super(AttrDict, self).__setitem__(key, value)

    def __getitem__(self, name):
        return super(AttrDict, self).__getitem__(name)

    def __delitem__(self, name):
        return super(AttrDict, self).__delitem__(name)

    __getattr__ = __getitem__
    __setattr__ = __setitem__

    def copy(self):
        return AttrDict(self)

    def print_config(self):
        def _flatten(d, prefix=""):
            rows = []
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    rows.extend(_flatten(v, key))
                else:
                    rows.append((key, v))
            return rows

        items = sorted(_flatten(self), key=lambda x: x[0])
        for i, (k, v) in enumerate(items):
            print(f"\033[0;33m{k}\033[0m: {v}, ", end="\n" if i % 5 == 4 else "")
        print()
