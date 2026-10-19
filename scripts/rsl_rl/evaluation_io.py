"""Read Isaac Lab's saved config snapshots, including explicitly allowed slices."""

from pathlib import Path

import yaml


class SnapshotLoader(yaml.FullLoader):
    pass


SnapshotLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice",
    lambda loader, node: slice(*loader.construct_sequence(node, deep=True)),
)


def load_snapshot(path):
    return yaml.load(Path(path).read_text(), Loader=SnapshotLoader)
