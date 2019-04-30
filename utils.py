import numpy as np
import tqdm
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
            self.interactions_dict = defaultdict(
                lambda: defaultdict(lambda: 0))
            for user, item, count in self.interactions:
                self.interactions_dict[user][item] = int(count)
        return self.interactions_dict


def auc(test_set, user_factors, subreddit_factors, subreddits, users):
    """
    Returns the auc score on a test data set
    """
    num_users = len(test_set)
    total = 0

    # treat the signal as 1 as per the implicit bpr paper
    for subreddit, user, signal in tqdm.tqdm_notebook(test_set):
        u = users_index[user]
        i = subreddits_index[subreddit]

        x_ui = user_factors[u].dot(subreddit_factors[i])

        js = []

        for j in range(0, num_subreddits):
            if j != i and j not in E_u[u]:
                js.append(j)

        total += np.sum(np.heaviside(x_ui - \
                        user_factors[u].dot(subreddit_factors[js].T), 0)) / len(js)

    return total / num_users
