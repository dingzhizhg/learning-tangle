import hashlib
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional

class Block:
    """区块链中的区块类，用于存储模型更新和验证信息"""
    
    def __init__(self, 
                 index: int,
                 previous_hash: str,
                 timestamp: float,
                 model_data: Dict[str, Any],
                 nonce: int = 0,
                 miner_id: str = None,
                 signature: str = None):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.model_data = model_data
        self.nonce = nonce
        self.miner_id = miner_id
        self.signature = signature
        self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """计算区块的哈希值"""
        block_string = json.dumps({
            'index': self.index,
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp,
            'model_data': self.model_data,
            'nonce': self.nonce,
            'miner_id': self.miner_id
        }, sort_keys=True, default=str)
        
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """将区块转换为字典格式"""
        return {
            'index': self.index,
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp,
            'model_data': self.model_data,
            'nonce': self.nonce,
            'miner_id': self.miner_id,
            'signature': self.signature,
            'hash': self.hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Block':
        """从字典创建区块"""
        block = cls(
            index=data['index'],
            previous_hash=data['previous_hash'],
            timestamp=data['timestamp'],
            model_data=data['model_data'],
            nonce=data['nonce'],
            miner_id=data.get('miner_id'),
            signature=data.get('signature')
        )
        block.hash = data['hash']
        return block
    
    def mine_block(self, difficulty: int) -> bool:
        """挖矿：寻找满足难度要求的nonce值"""
        target = '0' * difficulty
        
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
        
        return True
    
    def is_valid(self) -> bool:
        """验证区块的有效性"""
        return self.hash == self.calculate_hash() 