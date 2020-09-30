import os
import pathlib
import shutil
from collections import namedtuple, MutableMapping

Timestep = namedtuple("Timestep", "t observation action reward next_observation is_terminal info")


def mkdir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def rmdir(path):
    shutil.rmtree(path)


def subdirectories(path):
    try: return [subdir for subdir in os.listdir(path) if os.path.isdir(os.path.join(path, subdir))]
    except FileNotFoundError: return []


def files(path):
    return [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]


def flatten_dict(dictionary, parent_key='', separator='.'):
    items = []
    for key, value in dictionary.items():
        key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping): items.extend(flatten_dict(value, key, separator).items())
        else: items.append((key, value))
    return dict(items)


def print_dict(d, show_missing=True):
    # From tensorboard's official code
    """Prints a shallow dict to console.

    Args:
    d: Dict to print.
    show_missing: Whether to show keys with empty values.
    """
    for k, v in sorted(d.items()):
        if (not v) and show_missing:
            # No instances of the key, so print missing symbol.
            print('{} -'.format(k))
        elif isinstance(v, list):
            # Value is a list, so print each item of the list.
            print(k)
            for item in v:
                print('   {}'.format(item))
        elif isinstance(v, dict):
            # Value is a dict, so print each (key, value) pair of the dict.
            print(k)
            for kk, vv in sorted(v.items()):
                print('   {:<20} {}'.format(kk, vv))