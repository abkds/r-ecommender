import numpy as np


def item_to_index(things):
    index = {}
    for idx, item in enumerate(things):
        index[item] = idx
    return index


def index_to_item(index):
    things = np.empty(len(index), dtype=object)
    for item, idx in index.items():
        things[idx] = item
    return things
