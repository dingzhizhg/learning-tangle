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
    def reference_selection(self, num_tips, exclude_parent=None):
        entry_point = self.tangle.genesis
        
        reference = []
        used_tips = set()  # 用于跟踪已选择的tips
        
        # 如果指定了要排除的parent，将其加入已使用集合
        if exclude_parent:
            used_tips.add(exclude_parent)
        
        # 确保选择的tips数量不超过可用的tips数量
        available_tips = self.get_all_tips()
        # 排除已使用的tips
        available_tips = [tip for tip in available_tips if tip not in used_tips]
        
        if len(available_tips) < num_tips:
            num_tips = len(available_tips)
        
        attempts = 0
        max_attempts = num_tips * 20  # 增加最大尝试次数
        
        while len(reference) < num_tips and attempts < max_attempts:
            temp = self.randomwalk(entry_point, self.children_list)
            
            # 确保选择的tip不在已选择的列表中
            if temp not in used_tips and temp in available_tips:
                reference.append(temp)
                used_tips.add(temp)
            
            attempts += 1
        
        # 如果通过随机行走无法获得足够的unique tips，直接从可用tips中随机选择
        if len(reference) < num_tips:
            remaining_tips = [tip for tip in available_tips if tip not in used_tips]
            needed = num_tips - len(reference)
            if len(remaining_tips) >= needed:
                selected = random.sample(remaining_tips, needed)
                reference.extend(selected)
                used_tips.update(selected)
        
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
        
        # 使用与tangle/tip_selector.py相同的权重计算方法
        weights = self.ratings_to_weight(rating_list)
        
        # 使用加权随机选择
        return self.weighted_choice(children_list, weights)
    
    @staticmethod
    def weighted_choice(approvers, weights):
        """加权随机选择，参考tangle/tip_selector.py的实现"""
        rn = random.uniform(0, sum(weights))
        for i in range(len(approvers)):
            rn -= weights[i]
            if rn <= 0:
                return approvers[i]
        return approvers[-1]
    
    @staticmethod
    def ratings_to_weight(ratings):
        """将ratings转换为权重，参考tangle/tip_selector.py的实现"""
        highest_rating = max(ratings)
        normalized_ratings = [r - highest_rating for r in ratings]
        return [np.exp(r * ALPHA) for r in normalized_ratings]

    def get_all_tips(self):
        """获取所有可用的tips"""
        tips = []
        for tx in self.tangle.transactions:
            if len(self.children_list[tx]) == 0:  # 没有子节点的就是tip
                tips.append(tx)
        return tips
