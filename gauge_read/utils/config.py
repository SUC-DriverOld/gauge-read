import os

import torch
import yaml


class AttrDict(dict):
    """A dictionary with attribute-style access. It maps attribute access to
    the real dictionary.
    """

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

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


config = AttrDict()


def _to_attr_dict(value):
    if isinstance(value, dict):
        node = AttrDict()
        for k, v in value.items():
            node[k] = _to_attr_dict(v)
        return node
    if isinstance(value, list):
        return [_to_attr_dict(v) for v in value]
    return value


def _set_runtime_fields(cfg):
    cuda_enabled = bool(cfg.system.get("cuda", False))
    cfg.system.device = torch.device("cuda") if cuda_enabled else torch.device("cpu")


def _load_to_config(data):
    config.clear()
    for k, v in data.items():
        config[k] = _to_attr_dict(v)
    _set_runtime_fields(config)


def ensure_loaded(config_path=None):
    """Ensure global config is initialized.

    This keeps legacy call sites working where modules read ``config`` at import
    time (for example, WebUI startup).
    """
    if "system" in config:
        return

    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config.yaml")

    load_config(config_path)


def load_config(config_path):
    """Load YAML config file as nested AttrDict sections."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a top-level mapping.")

    _load_to_config(data)


def update_config(cfg, extra_config):
    for k, v in vars(extra_config).items():
        if k == "config" or v is None:
            continue

        # Fall back to top-level assignment for future extension.
        cfg[k] = v
    _set_runtime_fields(cfg)


def print_config(cfg):
    def _flatten(d, prefix=""):
        rows = []
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                rows.extend(_flatten(v, key))
            else:
                rows.append((key, v))
        return rows

    items = sorted(_flatten(cfg), key=lambda x: x[0])
    for i, (k, v) in enumerate(items):
        print(f"\033[0;33m{k}\033[0m: {v}, ", end="\n" if i % 5 == 4 else "")
    print()


# Auto-load default configuration for import-time consumers.
ensure_loaded()
