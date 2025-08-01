import random

import numpy as np
from .tip_selector import TipSelector


class MaliciousTipSelector(TipSelector):
    def compute_ratings(self, children_list):
        rating = {}
        future_set_cache = {}
        for tx in self.tangle.transactions:
            rating[tx] = int(self.tangle.transactions[tx].malicious)
            #rating[tx] = sum([self.tangle.transactions[future_tx].malicious for future_tx in
            #                  self.future_set(tx, children_list, future_set_cache)]) + (
            #                 self.tangle.transactions[tx].malicious)


        return rating