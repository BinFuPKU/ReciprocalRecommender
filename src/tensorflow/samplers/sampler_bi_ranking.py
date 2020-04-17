
import numpy as np
from threading import Thread
from multiprocessing import Queue

# user: posItem: n negItems
class Sampler(object):
    def __init__(self, A2B_trasR, B2A_trasR, n_neg=5, batch_size=100, n_workers=1):
        self.batch_size, self.n_neg = batch_size, n_neg

        self.n_A, self.n_B = A2B_trasR.shape

        # A2B
        self.A2B_user_posItemset = {}
        for u, row in enumerate(A2B_trasR.rows):
            if row:
                self.A2B_user_posItemset[u] = set(row)
        self.A_users = list(self.A2B_user_posItemset.keys())
        # B2A
        self.B2A_user_posItemset = {}
        for u, row in enumerate(B2A_trasR.rows):
            if row:
                self.B2A_user_posItemset[u] = set(row)
        self.B_users = list(self.B2A_user_posItemset.keys())

        self.result_queue = Queue(maxsize=n_workers*2)
        self.threads = []
        for i in range(n_workers):
            self.threads.append(Thread(target=self.__sample_function__))
            self.threads[-1].start()

    def __sample_function__(self,):
        while True:
            np.random.shuffle(self.A_users)
            np.random.shuffle(self.B_users)

            batch_num = min(int(len(self.A_users) / self.batch_size), int(len(self.B_users) / self.batch_size))

            for i in range(batch_num):
                # positive item samples
                A_batch = np.zeros((self.batch_size, 2+self.n_neg))
                B_batch = np.zeros((self.batch_size, 2+self.n_neg))

                A_batch[:, 0] = self.A_users[i * self.batch_size: (i + 1) * self.batch_size]
                B_batch[:, 0] = self.B_users[i * self.batch_size: (i + 1) * self.batch_size]

                for ind in range(A_batch.shape[0]):
                    A_user = int(A_batch[ind, 0])
                    A_batch[ind, 1] = np.random.choice(list(self.A2B_user_posItemset[A_user]))
                    A_batch[ind, 2:] = np.random.randint(low=0, high=self.n_B, size=(self.n_neg))

                    for j in range(self.n_neg):
                        negItem = np.random.randint(low=0, high=self.n_B)
                        while negItem in self.A2B_user_posItemset[A_user]:
                            negItem = np.random.randint(low=0, high=self.n_B)
                        A_batch[ind, 2+j] = negItem

                for ind in range(B_batch.shape[0]):
                    B_user = int(B_batch[ind, 0])
                    B_batch[ind, 1] = np.random.choice(list(self.B2A_user_posItemset[B_user]))
                    B_batch[ind, 2:] = np.random.randint(low=0, high=self.n_A, size=(self.n_neg))

                    for j in range(self.n_neg):
                        negItem = np.random.randint(low=0, high=self.n_A)
                        while negItem in self.B2A_user_posItemset[B_user]:
                            negItem = np.random.randint(low=0, high=self.n_A)
                        B_batch[ind, 2+j] = negItem

                self.result_queue.put((A_batch, B_batch))

    def next_batch(self):
        return self.result_queue.get()
