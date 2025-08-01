import os
import hashlib
from tempfile import TemporaryFile

import numpy as np

class Transaction:
    def __init__(self, weights, accuracy, parent, reference, id=None, tag=None, malicious=False):
        self.parent = parent
        self.reference = reference
        self.id = id
        self.weights = weights
        self.accuracy = accuracy
        self.tag = tag
        self.malicious = malicious

    def load_weights(self):
        if self.weights is None and self.id is not None:
            # data = np.load(f'tangle_data/transactions/{self.id}')
            self.weights = np.load(f'tangle_data/transactions/{self.id}.npy', allow_pickle=True)

        return self.weights

    def name(self):
        if self.id is None:
            with TemporaryFile() as tmpfile:
                self.save(tmpfile)
                tmpfile.seek(0)
                self.id = self.hash_file(tmpfile)

            with open(f'tangle_data/transactions/{self.id}.npy', 'wb') as tx_file:
                self.save(tx_file)
        return self.id

    @staticmethod
    def hash_file(f):
        BUF_SIZE = 65536
        sha1 = hashlib.sha1()
        while True:
          data = f.read(BUF_SIZE)
          if not data:
              break
          sha1.update(data)

        return sha1.hexdigest()

    def save(self, file):
      np.save(file, self.weights, allow_pickle=True)

    def add_tag(self, tag):
        self.tag = tag
