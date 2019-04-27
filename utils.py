import numpy as np
from collections import defaultdict


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


class Interaction():
    "Deal with basic manipulations of Interaction object"
    def __init__(self, interactions, num_users, num_items):
        self.interactions = interactions
        self.num_users = num_users
        self.num_items = num_items
        self.interactions_dict = None

    def get_interaction_dict(self):
        "Returns the interactions in the form of dictionary"
        if self.interactions_dict is None:
            self.interactions_dict = defaultdict(lambda: set())
            for user, item, _ in self.interactions:
                self.interactions_dict[user].add(item)
        return self.interactions_dict