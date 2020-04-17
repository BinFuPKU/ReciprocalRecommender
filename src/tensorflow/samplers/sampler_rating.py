
import numpy as np
from threading import Thread
from multiprocessing import Queue

# user: posItem: ratio negItems
class Sampler(object):
    def __init__(self, trasR, negRatio=1., batch_size=100, n_workers=1):
        self.batch_size, self.negRatio = batch_size, negRatio

        self.n_users, self.n_items = trasR.shape[0], trasR.shape[1]

        self.useritemrating_tuples = np.array([(user, item, trasR[user, item]) for user, item in np.asarray(trasR.nonzero()).T])
        self.user_posItemset = {u: set(row) for u, row in enumerate(trasR.rows)}

        self.result_queue = Queue(maxsize=n_workers*2)
        self.threads = []
        for i in range(n_workers):
            self.threads.append(Thread(target=self.__sample_function__))
            self.threads[-1].start()

    def __sample_function__(self,):
        while True:
            for i in range(int(len(self.useritemrating_tuples) / self.batch_size)):
                # positive item samples
                useritemrating_tuples_batch = self.useritemrating_tuples[i * self.batch_size: (i + 1) * self.batch_size, :]

                # negative item samples
                num_neg = int(self.batch_size*self.negRatio)
                if num_neg>0:
                    useritemrating_tuples_batch_neg = []
                    for _ in range(num_neg):
                        user, negitem = np.random.randint(0, self.n_users), np.random.randint(0, self.n_items)
                        while negitem in self.user_posItemset[user]:
                            negitem = np.random.randint(0, self.n_items)
                        useritemrating_tuples_batch_neg.append([user, negitem, 0])
                    useritemrating_tuples_batch = np.concatenate([useritemrating_tuples_batch, np.array(useritemrating_tuples_batch_neg)])
                np.random.shuffle(useritemrating_tuples_batch)
                self.result_queue.put(useritemrating_tuples_batch)

    def next_batch(self):
        return self.result_queue.get()

