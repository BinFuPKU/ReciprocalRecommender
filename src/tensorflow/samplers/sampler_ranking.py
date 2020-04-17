
import numpy as np
from threading import Thread
from multiprocessing import Queue

# user: posItem: n negItems
class Sampler(object):
    def __init__(self, n_A, n_B, trasR, n_neg=5, batch_size=100, n_workers=1):
        self.batch_size, self.n_neg = batch_size, n_neg

        self.n_A, self.n_B = n_A, n_B

        self.user_posItemset = {}
        for u, row in enumerate(trasR.rows):
            if row:
                self.user_posItemset[u] = set(row)
        self.users = list(self.user_posItemset.keys())

        self.result_queue = Queue(maxsize=n_workers*2)
        self.threads = []
        for i in range(n_workers):
            self.threads.append(Thread(target=self.__sample_function__))
            self.threads[-1].start()

    def __sample_function__(self,):
        while True:
            np.random.shuffle(self.users)
            for i in range(int(len(self.users) / self.batch_size)):
                # positive item samples
                batch = np.zeros((self.batch_size, 2+self.n_neg))

                batch[:, 0] = self.users[i * self.batch_size: (i + 1) * self.batch_size]

                for ind in range(batch.shape[0]):
                    user = int(batch[ind, 0])
                    batch[ind, 1] = np.random.choice(list(self.user_posItemset[user]))
                    batch[ind, 2:] = np.random.randint(low=0, high=self.n_A + self.n_B, size=(self.n_neg))

                    negItem_scope = [0, self.n_B] if user < self.n_A else [self.n_B, self.n_A + self.n_B]

                    for j in range(self.n_neg):
                        negItem = np.random.randint(low=negItem_scope[0], high=negItem_scope[1])
                        while negItem in self.user_posItemset[user]:
                            negItem = np.random.randint(low=negItem_scope[0], high=negItem_scope[1])
                        batch[ind, 2+j] = negItem

                self.result_queue.put(batch)

    def next_batch(self):
        return self.result_queue.get()
