
import numpy as np
from threading import Thread
from multiprocessing import Queue

# user: posItem: n negItems
class Sampler(object):
    def __init__(self, n_A, n_B, trasR, n_neg=5, batch_size=100, n_workers=1):
        self.batch_size, self.n_neg = batch_size, n_neg

        self.n_A, self.n_B = n_A, n_B

        self.useritem_pairs = np.array(trasR.nonzero()).T
        self.user_posItemset = {u: set(row) for u, row in enumerate(trasR.rows)}

        self.result_queue = Queue(maxsize=n_workers*2)
        self.threads = []
        for i in range(n_workers):
            self.threads.append(Thread(target=self.__sample_function__))
            self.threads[-1].start()

    def __sample_function__(self,):
        while True:
            np.random.shuffle(self.useritem_pairs)
            for i in range(int(len(self.useritem_pairs) / self.batch_size)):
                # positive item samples
                useritem_pairs_batch = self.useritem_pairs[i * self.batch_size: (i + 1) * self.batch_size, :]

                # negative item samples
                negItems_batch = np.random.randint(low=0, high=self.n_A + self.n_B, size=(useritem_pairs_batch.shape[0], self.n_neg))

                for i in range(useritem_pairs_batch.shape[0]):
                    user = useritem_pairs_batch[i, 0]
                    negItem_scope = [0, self.n_B] if user < self.n_A else [self.n_B, self.n_A + self.n_B]

                    for j in range(self.n_neg):
                        negItem = np.random.randint(low=negItem_scope[0], high=negItem_scope[1])
                        while negItem in self.user_posItemset[user]:
                            negItem = np.random.randint(low=negItem_scope[0], high=negItem_scope[1])
                        negItems_batch[i, j] = negItem

                data = np.hstack((useritem_pairs_batch, negItems_batch))
                self.result_queue.put(data)

    def next_batch(self):
        return self.result_queue.get()
