import numpy as np


class DnS(object):

    def __init__(self):
        self.name = 'DnS'
        self.videos = set(np.loadtxt('data/dns_100k.txt', dtype=str).tolist())

    def get_queries(self):
        return []

    def get_database(self):
        return sorted(list(self.videos))

    def evaluate(self, similarities, all_db=None):
        raise NotImplemented('DnS-100K dataset is used only for training. No annotations are provided')
