import itertools
import random

import numpy as np

class TipSelector:
    def __init__(self, tangle):
        self.tangle = tangle

        # Build a map of transactions that are children of a given transaction
        self.children_list = {x: [] for x in self.tangle.transactions}
        for x, tx in self.tangle.transactions.items():
            if tx.parent is not None:
                self.children_list[tx.parent].append(x)

        self.weights = self.compute_weights(self.children_list)

        accuracies = {}
        for x, tx in self.tangle.transactions.items():
            accuracies[x] = tx.accuracy
        self.accuracies = accuracies

    # parent selection
    def parent_selection(self):
        entry_point = self.tangle.genesis

        parent = self.walk(entry_point, self.weights, self.children_list)

        return parent

    def compute_weights(self, children_list):
        weights = {}
        future_set_cache = {}
        for tx in self.tangle.transactions:
            weights[tx] = len(self.future_set(tx, children_list, future_set_cache)) + 1

        return weights

    def future_set(self, tx, children_list, future_set_cache):
        def recurse_future_set(t):
            if t not in future_set_cache:
                direct_approvals = set(children_list[t])
                future_set_cache[t] = direct_approvals.union(*[recurse_future_set(x) for x in direct_approvals])

            return future_set_cache[t]

        return recurse_future_set(tx)

    def walk(self, tx, weights, children_list):
        step = tx
        prev_step = None

        while step:
            children = children_list[step]
            prev_step = step
            step = self.next_step(weights, children)

        return prev_step

    def next_step(self, weights, children):
        # 判断是否为 tip tx
        if len(children) == 0:
            return None

        children_weights = [weights[a] for a in children]
        
        max_weight_index = children_weights.index(max(children_weights))
        step = children[max_weight_index]

        return step
    
    # reference selection
    def reference_selection(self, num_tips, parent):
        entry_point = self.tangle.genesis
        
        reference = []
        for i in range(num_tips-1):
            temp = self.randomwalk(entry_point, self.accuracies, self.children_list)
            reference.append(temp)

        return reference
    
    def randomwalk(self, tx, accuracies, children_list):
        step = tx
        prev_step = None

        while step:
            children = children_list[step]
            prev_step = step
            step = self.random_step(accuracies, children)
        
        return prev_step
    
    def random_step(self, accuracies, children):
        if len(children) == 0:
            return None
        
        # 过滤掉 accuracy 为 None 的 child
        valid_children = [a for a in children if accuracies[a] is not None]
        if not valid_children:
            return None
        children_accuracies = [accuracies[a] for a in valid_children]

        exp_accuracies = np.exp(children_accuracies)
        probabilities = exp_accuracies / np.sum(exp_accuracies)
        step = np.random.choice(valid_children, p=probabilities)

        return step
