import itertools
import random

import numpy as np

# https://docs.iota.org/docs/node-software/0.1/iri/references/iri-configuration-options
LAMBDA = 0.7
ALPHA = 0.001

class TipSelector:
    def __init__(self, tangle):
        self.tangle = tangle

        # Build a map of transactions that directly approve a given transaction
        self.approving_transactions = {x: [] for x in self.tangle.transactions}
        self.dirapproving_transactions = {x: [] for x in self.tangle.transactions}
        for x, tx in self.tangle.transactions.items():
            # 计算approving_transactions
            for unique_parent in tx.parents:
                self.approving_transactions[unique_parent].append(x)  
            # 计算dirapproving_transactions
            if len(list(tx.parents)) > 0:
                unique_parent = list(tx.parents)[0]
                self.dirapproving_transactions[unique_parent].append(x)

        self.get_depth(tangle.transactions)
        self.get_brother(tangle.transactions)

        self.ratings = self.compute_ratings(self.approving_transactions)

    def tip_selection(self, num_tips):
        entry_point = self.tangle.genesis

        tips = []
        for i in range(num_tips):
            tips.append(self.walk(entry_point, self.ratings, self.approving_transactions))

        return tips


    # 计算出每个tx的depth和brother的参数值
    def get_depth(self, transactions):
        for name, tx in transactions.items():
            #若没有parents，则深度为1
            if len(list(tx.parents)) == 0:
                tx.depth = 1
            #若有父交易，且深度未被定义，则深度为父交易的深度+1
            elif tx.depth == 0:
                to_parent = self.tangle.transactions[list(tx.parents)[0]]
                tx.depth = to_parent.depth + 1
            #若有父交易，且深度已被定义，则不进行操作

    def get_brother(self, transactions):
        # 计算每个tx的child数量
        for name, tx in transactions.items():
            tx.child = len(self.dirapproving_transactions[name])

        for name, tx in transactions.items():
            #若没有parents，则没有兄弟
            if len(list(tx.parents)) == 0:
                tx.brother = 0
            #若有父交易，则兄弟数量为父交易的child数量
            else:
                to_parent = self.tangle.transactions[list(tx.parents)[0]]
                tx.brother = to_parent.child

    def compute_ratings(self, approving_transactions):
        rating = {}
        future_set_cache = {}
        for tx in self.tangle.transactions:
            rating[tx] = len(self.future_set(tx, approving_transactions, future_set_cache)) + 1

        return rating
    

    def future_set(self, tx, approving_transactions, future_set_cache):
        def recurse_future_set(t):
            if t not in future_set_cache:
                direct_approvals = set(approving_transactions[t])
                future_set_cache[t] = direct_approvals.union(*[recurse_future_set(x) for x in direct_approvals])

            return future_set_cache[t]

        return recurse_future_set(tx)
    

    def walk(self, tx, ratings, approving_transactions):
        step = tx
        prev_step = None

        while step:
            approvers = approving_transactions[step]
            prev_step = step
            step = self.next_step(ratings, approvers)

        # When there are no more steps, this transaction is a tip
        return prev_step

    def next_step(self, ratings, approvers):
        approvers_with_rating = approvers  # There is a rating for every possible approver

        # There is no valid approver, this transaction is a tip
        if len(approvers_with_rating) == 0:
            return None

        approvers_ratings = [ratings[a] for a in approvers_with_rating]
        weights = self.ratings_to_weight(approvers_ratings)
        approver = self.weighted_choice(approvers_with_rating, weights)

        # Skip validation.
        # At least a validation of some PoW is necessary in a real-world implementation.

        return approver

        # if approver is not None:
        #     tail = validator.findTail(approver)
        #
        #     # If the selected approver is invalid, step back and try again
        #     if validator.isInvalid(tail):
        #         approvers = approvers.remove(approver)
        #
        #         return self.next_step(ratings, approvers)
        #
        #     return tail
        #
        # return None

    @staticmethod
    def weighted_choice(approvers, weights):
        # Instead of a random choice, one could also think about a more 'intelligent'
        # variant for this use case. E.g. choose a transaction that was published by a
        # node with 'similar' characteristics

        rn = random.uniform(0, sum(weights))
        for i in range(len(approvers)):
            rn -= weights[i]
            if rn <= 0:
                return approvers[i]
        return approvers[-1]

    @staticmethod
    def ratings_to_weight(ratings):
        highest_rating = max(ratings)
        normalized_ratings = [r - highest_rating for r in ratings]
        return [np.exp(r * ALPHA) for r in normalized_ratings]

    @staticmethod
    def ratings_to_probability(ratings):
        # Calculating a probability according to the IOTA randomness blog
        # https://blog.iota.org/alpha-d176d7601f1c
        b = sum(map(lambda r: np.exp(ALPHA * r),ratings))
        return [np.exp(r * ALPHA) / b for r in ratings]
