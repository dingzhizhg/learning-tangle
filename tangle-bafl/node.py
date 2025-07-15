import numpy as np
import tensorflow as tf
import sys
import time
import hashlib
from typing import Dict, Any, Optional, List

from .malicious_tip_selector import MaliciousTipSelector
from .poison_type import PoisonType
from .block import Block
from .blockchain import Blockchain

SELECTED_TIPS = 2

class ModelValidator:
    
    def __init__(self):
        self.validation_history = []
        self.malicious_models = set()
    
    def validate_local_model(self, model_data: Dict[str, Any]) -> bool:
        """验证本地模型"""
        required_fields = ['weights', 'score', 'device_id']
        for field in required_fields:
            if field not in model_data:
                return False
        
        if not self._validate_weights(model_data['weights']):
            return False
        
        if not self._validate_score(model_data['score']):
            return False
        
        if self._is_malicious_model(model_data):
            self.malicious_models.add(model_data['device_id'])
            return False
        
        return True
    
    def validate_global_model(self, global_model_data: Dict[str, Any]) -> bool:
        """验证全局模型"""
        required_fields = ['weights', 'score', 'device_id']
        for field in required_fields:
            if field not in global_model_data:
                return False
        
        if not self._validate_weights(global_model_data['weights']):
            return False
        
        if not self._validate_score(global_model_data['score']):
            return False
        
        return True
    
    def _validate_weights(self, weights) -> bool:
        """验证模型权重"""
        if weights is None:
            return False
        
        if not isinstance(weights, (list, np.ndarray)):
            return False
        
        for weight in weights:
            if weight is not None:
                if not isinstance(weight, (np.ndarray, list)):
                    return False
                
                if isinstance(weight, np.ndarray):
                    if not np.isfinite(weight).all():
                        return False
        
        return True
    
    def _validate_score(self, score) -> bool:
        """验证模型分数"""
        if not isinstance(score, (int, float, np.number)):
            return False
        
        if not np.isfinite(score):
            return False
        
        if score < 0 or score > 100:
            return False
        
        return True
    
    def _is_malicious_model(self, model_data: Dict[str, Any]) -> bool:
        """检测恶意模型"""
        weights = model_data['weights']
        if weights is None:
            return False
        
        for weight in weights:
            if weight is not None:
                if isinstance(weight, np.ndarray):
                    if np.any(np.abs(weight) > 1000):
                        return True
                    if np.all(weight == 0):
                        return True
                    if not np.isfinite(weight).all():
                        return True
        
        score = model_data['score']
        if score < 0.01 or score > 99.99:
            return True
        
        return False

class Node:
    def __init__(self, client, tangle, poison_type=PoisonType.NONE, blockchain=None, node_id=None):
        self.client = client
        self.tangle = tangle
        self.poison_type = poison_type
        self.blockchain = blockchain
        self.node_id = node_id or f"node_{id(self)}"
        self.validator = ModelValidator()
        self.pending_models = []

  def choose_tips(self, num_tips=2, sample_size=2, selector=None):
      if selector is None:
          selector = TipSelector(self.tangle)

      if len(self.tangle.transactions) < num_tips:
          return [self.tangle.transactions[self.tangle.genesis] for i in range(SELECTED_TIPS)]

      if self.poison_type != PoisonType.NONE:
          sample_size = num_tips

      tips = selector.tip_selection(sample_size)

      no_dups = set(tips)
      if len(no_dups) >= num_tips:
          tips = no_dups

      tip_txs = [self.tangle.transactions[tip] for tip in tips]


      # Find best tips
      if num_tips < sample_size:
          #   print(len(list(filter(lambda tip: tip.malicious, tip_txs))), "of", len(tip_txs), "found tips were malicous")
          # Choose tips with lowest test loss
          tip_losses = []
          loss_cache = {}
          for tip in tip_txs:
              if tip.id in loss_cache.keys():
                  tip_losses.append((tip, loss_cache[tip.id]))
              else:
                  self.client.model.set_params(tip.load_weights())
                  loss = self.client.test('test')['loss']
                  tip_losses.append((tip, loss))
                  loss_cache[tip.id] = loss
          best_tips = sorted(tip_losses, key=lambda tup: tup[1], reverse=False)[:num_tips]
          tip_txs = [tup[0] for tup in best_tips]
      #   print(len(list(filter(lambda tip: tip.malicious, tip_txs))), "of", len(tip_txs), "selected tips were malicous")

      return tip_txs

  def compute_confidence(self, selector=None, approved_transactions_cache={}):
      num_sampling_rounds = 35

      transaction_confidence = {x: 0 for x in self.tangle.transactions}

      def approved_transactions(transaction):
          if transaction not in approved_transactions_cache:
              result = set([transaction]).union(*[approved_transactions(parent) for parent in self.tangle.transactions[transaction].parents])
              approved_transactions_cache[transaction] = result

          return approved_transactions_cache[transaction]

      # Use a cached tip selector
      if selector is None:
          selector = TipSelector(self.tangle)

      for i in range(num_sampling_rounds):
          tips = self.choose_tips(selector=selector)
          for tip in tips:
              for tx in approved_transactions(tip.name()):
                  transaction_confidence[tx] += 1

      return {tx: float(transaction_confidence[tx]) / (num_sampling_rounds * 2) for tx in self.tangle.transactions}

  def compute_cumulative_score(self, transactions, approved_transactions_cache={}):
      def compute_approved_transactions(transaction):
          if transaction not in approved_transactions_cache:
              result = set([transaction]).union(*[compute_approved_transactions(parent) for parent in self.tangle.transactions[transaction].parents])
              approved_transactions_cache[transaction] = result

          return approved_transactions_cache[transaction]

      return {tx: len(compute_approved_transactions(tx)) for tx in transactions}

  def compute_poisoning_score(self, transactions, approved_transactions_cache={}):
      def compute_approved_transactions(transaction):
          if transaction not in approved_transactions_cache:
              result = set([transaction]).union(*[compute_approved_transactions(parent) for parent in self.tangle.transactions[transaction].parents])
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
    
    # 区块链相关方法
    def update_local_model(self, num_epochs, batch_size):
        """步骤1：本地模型更新"""
        comp, num_samples, update = self.client.train(num_epochs, batch_size)
        local_weights = self.client.model.get_params()
        local_score = self.client.test('test')['loss']
        
        return {
            'weights': local_weights,
            'score': local_score,
            'device_id': self.node_id,
            'timestamp': time.time(),
            'type': 'local_model'
        }
    
    def upload_local_model(self, model_data):
        """步骤2：本地模型上传"""
        if self.blockchain:
            # 验证模型
            if self.validator.validate_local_model(model_data):
                # 添加到区块链待处理队列
                self.blockchain.add_pending_model(model_data)
                return True
        return False
    
    def download_global_model_and_score(self):
        """步骤4：模型和分数下载"""
        if self.blockchain:
            global_model = self.blockchain.get_global_model()
            device_history = self.blockchain.get_model_history(self.node_id)
            current_score = device_history[-1]['score'] if device_history else 0.0
            
            return {
                'global_model': global_model,
                'current_score': current_score,
                'device_history': device_history
            }
        return None
    
    def upload_global_model_and_score(self, global_model_data, new_score):
        """步骤6：全局模型和分数上传"""
        if self.blockchain:
            if self.validator.validate_global_model(global_model_data):
                model_data = {
                    'type': 'global_model',
                    'weights': global_model_data['weights'],
                    'score': new_score,
                    'device_id': self.node_id,
                    'timestamp': time.time(),
                    'aggregation_round': global_model_data.get('aggregation_round', 0)
                }
                self.blockchain.add_pending_model(model_data)
                return True
        return False
    
    def mine_block(self):
        """步骤8：区块生成"""
        if self.blockchain and self.blockchain.pending_models:
            return self.blockchain.mine_pending_models(self.node_id)
        return None
    
    def propagate_block(self, block):
        """步骤9：区块传播"""
        if block and self.blockchain:
            if self.blockchain.add_block(block):
                self.blockchain.broadcast_block(block)
                return True
        return False
    
    def download_latest_global_model(self):
        """步骤10：全局模型下载"""
        if self.blockchain:
            return self.blockchain.get_global_model()
        return None

  def process_next_batch(self, num_epochs, batch_size, num_tips=2, sample_size=2, reference_avg_top=1):
    # if self.poison_type == PoisonType.NONE:
    selector = TipSelector(self.tangle)
    # else:
    #     selector = MaliciousTipSelector(self.tangle)

    # Compute reference metrics
    reference_txs, reference, _ = self.obtain_reference_params(avg_top=reference_avg_top, selector=selector)
    self.client.model.set_params(reference)
    c_metrics = self.client.test('test')

    # Obtain number of tips from the tangle
    tips = self.choose_tips(num_tips=num_tips, sample_size=sample_size, selector=selector)

    if self.poison_type == PoisonType.RANDOM:
        weights = self.client.model.get_params()
        malicious_weights = [np.random.RandomState().normal(size=w.shape) for w in weights]
        print('generated malicious weights')
        return Transaction(malicious_weights, set([tip.name() for tip in tips]), malicious=True), None, None
    elif self.poison_type == PoisonType.LABELFLIP:
        # Todo Choose tips or reference model?
        averaged_weights = self.average_model_params(*[tip.load_weights() for tip in tips])
        self.client.model.set_params(averaged_weights)
        self.client.train(num_epochs, batch_size)
        print('trained on label-flip data')
        return Transaction(self.client.model.get_params(), set([tip.name() for tip in tips]), malicious=True), None, None
    else:
        # Perform averaging

        # How averaging is done exactly (e.g. weighted, using which weights) is left to the
        # network participants. It is not reproducible or verifiable by other nodes because
        # only the resulting weights are published.
        # Once a node has published its training results, it thus can't be sure if
        # and by what weight its delta is being incorporated into approving transactions.
        # However, assuming most nodes are well-behaved, they will make sure that eventually
        # those weights will prevail that incorporate as many partial results as possible
        # in order to prevent over-fitting.

        # Here: simple unweighted average
        averaged_weights = self.average_model_params(*[tip.load_weights() for tip in tips])
        self.client.model.set_params(averaged_weights)
        comp, num_samples, update = self.client.train(num_epochs, batch_size)

        c_averaged_model_metrics = self.client.test('test')
        if c_averaged_model_metrics['loss'] < c_metrics['loss']:
            return Transaction(self.client.model.get_params(), set([tip.name() for tip in tips])), c_averaged_model_metrics, comp

    return None, None, None
