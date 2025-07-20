import numpy as np
import tensorflow as tf
import sys

from .tip_selector import TipSelector
from .malicious_tip_selector import MaliciousTipSelector
from .transaction import Transaction
from .poison_type import PoisonType

NUM_TIPS = 2
LAMBDA = 0.5

class Node:
  def __init__(self, client, tangle, poison_type=PoisonType.NONE):
    self.client = client
    self.tangle = tangle
    self.poison_type = poison_type 

  def choose_tips(self, num_tips=NUM_TIPS, selector=None):
      if selector is None:
          selector = TipSelector(self.tangle)

    #   if len(self.tangle.transactions) < num_tips:
    #     parent = self.tangle.transactions[self.tangle.genesis]
    #     reference = [self.tangle.transactions[self.tangle.genesis] for i in range(num_tips)]
    #     return parent, reference

      parent_key = selector.parent_selection()
      reference_keys = selector.reference_selection(num_tips)
      
      # 确保parent不在reference列表中
      if parent_key in reference_keys:
          # 如果parent在reference中，从reference中移除并重新选择一个
          reference_keys.remove(parent_key)
        #   # 获取额外的reference来补充
        #   additional_reference = selector.reference_selection(1)
        #   if additional_reference and additional_reference[0] not in reference_keys:
        #       reference_keys.append(additional_reference[0])
      
      # 确保reference列表中没有重复项
      reference_keys = list(set(reference_keys))
      
    #   # 如果去重后reference数量不足，补充更多
    #   while len(reference_keys) < num_tips:
    #       additional_reference = selector.reference_selection(1)
    #       if additional_reference and additional_reference[0] not in reference_keys:
    #           reference_keys.append(additional_reference[0])
    #       else:
    #           # 如果无法获取更多unique的reference，跳出循环避免无限循环
    #           break

      parent = self.tangle.transactions[parent_key]
      reference = [self.tangle.transactions[k] for k in reference_keys]

      return parent, reference

  def compute_confidence(self, selector=None, approved_transactions_cache={}):
      num_sampling_rounds = 35

      transaction_confidence = {x: 0 for x in self.tangle.transactions}

      def approved_transactions(transaction):
          if transaction not in approved_transactions_cache:
              result = set([transaction]).union(*[approved_transactions(r) for r in self.tangle.transactions[transaction].reference])
              approved_transactions_cache[transaction] = result

          return approved_transactions_cache[transaction]

      # Use a cached tip selector
      if selector is None:
          selector = TipSelector(self.tangle)

      for i in range(num_sampling_rounds):
          parent, reference = self.choose_tips(selector=selector)
          tips = list(reference)
          tips.append(parent)
          for tip in tips:
              for tx in approved_transactions(tip.name()):
                  transaction_confidence[tx] += 1

      return {tx: float(transaction_confidence[tx]) / (num_sampling_rounds * 2) for tx in self.tangle.transactions}

  def compute_cumulative_score(self, transactions, approved_transactions_cache={}):
      def compute_approved_transactions(transaction):
          if transaction not in approved_transactions_cache:
              result = set([transaction]).union(*[compute_approved_transactions(r) for r in self.tangle.transactions[transaction].reference])
              approved_transactions_cache[transaction] = result

          return approved_transactions_cache[transaction]

      return {tx: len(compute_approved_transactions(tx)) for tx in transactions}

  def compute_poisoning_score(self, transactions, approved_transactions_cache={}):
      def compute_approved_transactions(transaction):
          if transaction not in approved_transactions_cache:
              result = set([transaction]).union(*[compute_approved_transactions(r) for r in self.tangle.transactions[transaction].reference])
              approved_transactions_cache[transaction] = result

          return approved_transactions_cache[transaction]

      return {tx: int(self.tangle.transactions[tx].malicious) + sum([self.tangle.transactions[transaction].malicious for transaction in compute_approved_transactions(tx)]) for tx in transactions}

  def obtain_reference_params(self, avg_top=1, selector=None):
      # Establish the 'current best'/'reference' weights from the tangle

      approved_transactions_cache = {}

      # 1. Perform tip selection n times, establish confidence for each transaction
      # (i.e. which transactions were already approved by most of the current tips?)
      transaction_confidence = self.compute_confidence(selector=selector, approved_transactions_cache=approved_transactions_cache)

      # 2. Compute cumulative score for transactions
      # (i.e. how many other transactions does a given transaction indirectly approve?)
      keys = [x for x in self.tangle.transactions]
      scores = self.compute_cumulative_score(keys, approved_transactions_cache=approved_transactions_cache)

      # How many directly or indirectly approved transactions are poisonous
      poison_scores = self.compute_poisoning_score(keys, approved_transactions_cache=approved_transactions_cache)
      poison_percentages = {tx: poison_scores[tx]/scores[tx] for tx in keys}

      # 3. For the top 100 transactions, compute the average
      best = sorted(
          {tx: scores[tx] * transaction_confidence[tx] for tx in keys}.items(),
          key=lambda kv: kv[1], reverse=True
      )[:avg_top]
      reference_txs = [elem[0] for elem in best]
      reference_params = self.average_model_params(*[self.tangle.transactions[elem].load_weights() for elem in reference_txs])
      reference_poison_score = np.mean([poison_percentages[elem] for elem in reference_txs])
      return reference_txs, reference_params, reference_poison_score

  def average_model_params(self, *params):
    return sum(params) / len(params)

  def process_next_batch(self, num_epochs, batch_size, num_tips=NUM_TIPS, reference_avg_top=1):
    selector = TipSelector(self.tangle)

    # Compute reference metrics
    reference_txs, reference, _ = self.obtain_reference_params(avg_top=reference_avg_top, selector=selector)
    self.client.model.set_params(reference)
    c_metrics = self.client.test('test')

    # Obtain number of tips from the tangle
    parent, reference_tip = self.choose_tips(num_tips=num_tips, selector=selector)
    tips = list(reference_tip)
    tips.append(parent)

    if self.poison_type == PoisonType.RANDOM:
        weights = self.client.model.get_params()
        malicious_weights = [np.random.RandomState().normal(size=w.shape) for w in weights]
        print('generated malicious weights')
        return Transaction(malicious_weights, None, parent.name(), [r.name() for r in reference], malicious=True), None, None
    elif self.poison_type == PoisonType.LABELFLIP:
        averaged_weights = self.average_model_params(*[tip.load_weights() for tip in tips])
        self.client.model.set_params(averaged_weights)
        self.client.train(num_epochs, batch_size)
        print('trained on label-flip data')
        return Transaction(self.client.model.get_params(), None, parent.name(), [r.name() for r in reference], malicious=True), None, None
    else:
        averaged_weights = self.average_model_params(*[tip.load_weights() for tip in tips])
        self.client.model.set_params(averaged_weights)
        comp, num_samples, update = self.client.train(num_epochs, batch_size)
        c_averaged_model_metrics = self.client.test('test')
        if c_averaged_model_metrics['loss'] < c_metrics['loss']:
            return Transaction(self.client.model.get_params(), c_averaged_model_metrics['accuracy'], parent.name(), [r.name() for r in reference_tip]), c_averaged_model_metrics, comp

    return None, None, None
