import itertools
import random

import numpy as np

ALPHA = 0.001

class TipSelector:
    def __init__(self, tangle):
        self.tangle = tangle

        # Build a map of transactions that are children of a given transaction
        self.children_list = {x: [] for x in self.tangle.transactions}
        for x, tx in self.tangle.transactions.items():
            if tx.parent is not None:
                self.children_list[tx.parent].append(x)

        self.weights = self.compute_weights(self.children_list)

        accuracies, ratings = {}, {}
        for x, tx in self.tangle.transactions.items():
            if tx.accuracy is not None:
                accuracies[x] = tx.accuracy
                ratings[x] = self.weights[x] * np.exp(tx.accuracy)
            else:
                accuracies[x] = 0
                ratings[x] = 0
        self.accuracies = accuracies
        self.ratings = ratings

        # self.ratings = [w * np.exp(a) for w, a in zip(self.weights, self.accuracies)]

        # 依照公式计算 ratings
        # 这个公式就是调整的重点！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        

        # highest_weight = max(weights)
        # normalized_ratings = [r - highest_weight for r in weights]
        # ratings = [np.exp(r * ALPHA) for r in normalized_ratings]

    # parent selection
    def parent_selection(self):
        entry_point = self.tangle.genesis

        parent = self.walk(entry_point, self.children_list)

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

    def walk(self, tx, children_list):
        step = tx
        prev_step = None

        while step:
            children = children_list[step]
            prev_step = step
            step = self.next_step(children)

        return prev_step

    def next_step(self, children_list):
        # 判断是否为 tip tx
        if len(children_list) == 0:
            return None

        children_ratings = [self.ratings[child] for child in children_list]
        
        max_weight_index = children_ratings.index(max(children_ratings))
        step = children_list[max_weight_index]

        return step
    
    # reference selection
    def reference_selection(self, num_tips):
        entry_point = self.tangle.genesis
        
        reference = []
        for i in range(num_tips-1):
            temp = self.randomwalk(entry_point, self.children_list)
            reference.append(temp)

        return reference
    
    def randomwalk(self, tx, children_list):
        step = tx
        prev_step = None

        while step:
            children = children_list[step]
            prev_step = step
            step = self.random_step(children)

        return prev_step
    
    def random_step(self, children_list):
        if len(children_list) == 0:
            return None

        # 遍历得到 rating_list
        rating_list = [self.ratings[child] for child in children_list]

        # 生成随机数，以依权重随机选择 child
        rn = random.uniform(0, sum(rating_list))
        for i in range(len(children_list)):
            rn -= rating_list[i]
            if rn <= 0:
                return children_list[i]
            
        return children_list[-1]
