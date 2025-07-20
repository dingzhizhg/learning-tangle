import numpy as np
import tensorflow as tf
import sys

from .tip_selector import TipSelector
from .malicious_tip_selector import MaliciousTipSelector
from .transaction import Transaction
from .poison_type import PoisonType

NUM_TIPS = 3
LAMBDA = 0.5

class Node:
  def __init__(self, client, tangle, poison_type=PoisonType.NONE):
    self.client = client
    self.tangle = tangle
    self.poison_type = poison_type 

  def choose_tips(self, num_tips=NUM_TIPS, selector=None):
      if selector is None:
          selector = TipSelector(self.tangle)

      if len(self.tangle.transactions) < num_tips:
        parents = [self.tangle.transactions[self.tangle.genesis] for i in range(num_tips)]
        return parents

      parents_key = selector.tip_selection(num_tips)
      parents = [self.tangle.transactions[k] for k in parents_key]

      return parents

  def compute_confidence(self, selector=None, approved_transactions_cache={}):
      num_sampling_rounds = 35

      transaction_confidence = {x: 0 for x in self.tangle.transactions}

      def approved_transactions(transaction):
          if transaction not in approved_transactions_cache:
              result = set([transaction]).union(*[approved_transactions(r) for r in self.tangle.transactions[transaction].parents])
              approved_transactions_cache[transaction] = result

          return approved_transactions_cache[transaction]

      # Use a cached tip selector
      if selector is None:
          selector = TipSelector(self.tangle)

      for i in range(num_sampling_rounds):
          parents = self.choose_tips(selector=selector)

          for tip in parents:
              for tx in approved_transactions(tip.name()):
                  transaction_confidence[tx] += 1

      return {tx: float(transaction_confidence[tx]) / (num_sampling_rounds * 2) for tx in self.tangle.transactions}

  def compute_cumulative_score(self, transactions, approved_transactions_cache={}):
      def compute_approved_transactions(transaction):
          if transaction not in approved_transactions_cache:
              result = set([transaction]).union(*[compute_approved_transactions(r) for r in self.tangle.transactions[transaction].parents])
              approved_transactions_cache[transaction] = result

          return approved_transactions_cache[transaction]

      return {tx: len(compute_approved_transactions(tx)) for tx in transactions}

  def compute_poisoning_score(self, transactions, approved_transactions_cache={}):
      def compute_approved_transactions(transaction):
          if transaction not in approved_transactions_cache:
              result = set([transaction]).union(*[compute_approved_transactions(r) for r in self.tangle.transactions[transaction].parents])
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
    parents = self.choose_tips(num_tips=num_tips, selector=selector)

    if self.poison_type == PoisonType.RANDOM:
        weights = self.client.model.get_params()
        malicious_weights = [np.random.RandomState().normal(size=w.shape) for w in weights]
        print('generated malicious weights')
        return Transaction(malicious_weights, None, [r.name() for r in parents], malicious=True), None, None
    elif self.poison_type == PoisonType.LABELFLIP:
        averaged_weights = self.average_model_params(*[tip.load_weights() for tip in parents])
        self.client.model.set_params(averaged_weights)
        self.client.train(num_epochs, batch_size)
        print('trained on label-flip data')
        return Transaction(self.client.model.get_params(), None, [r.name() for r in parents], malicious=True), None, None
    elif self.poison_type == PoisonType.LAZY:
        # Lazy attack: node does not train, just returns averaged weights without training
        averaged_weights = self.average_model_params(*[tip.load_weights() for tip in parents])
        print('lazy attack: no training performed, returning averaged weights')
        return Transaction(averaged_weights, None, [r.name() for r in parents], malicious=True), None, None
    else:
        averaged_weights = self.average_model_params(*[tip.load_weights() for tip in parents])
        self.client.model.set_params(averaged_weights)
        comp, num_samples, update = self.client.train(num_epochs, batch_size)
        c_averaged_model_metrics = self.client.test('test')
        if c_averaged_model_metrics['loss'] < c_metrics['loss']:
            return Transaction(self.client.model.get_params(), c_averaged_model_metrics['accuracy'], [r.name() for r in parents]), c_averaged_model_metrics, comp

    return None, None, None
