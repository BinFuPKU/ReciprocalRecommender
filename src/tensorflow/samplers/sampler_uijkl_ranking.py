
import numpy as np
from threading import Thread
from queue import Queue
from scipy.sparse import lil_matrix

# user: posItem: n negItems
class Sampler(object):
    def __init__(self, A2B_trasR, B2A_trasR, batch_size=100, n_workers=1):
        self.batch_size = batch_size

        A2B_match_trasR = lil_matrix(A2B_trasR.multiply(B2A_trasR.T).toarray())
        self.As = list(set(A2B_trasR.nonzero()[0]))
        self.n_A, self.n_B = A2B_trasR.shape[0], A2B_trasR.shape[1]

        self.A2B_uni_pairs = np.array(A2B_trasR.nonzero()).T
        self.A2B_uni_Bs = {u: set(row) for u, row in enumerate(A2B_trasR.rows)}

        self.A2B_bi_pairs = np.array(A2B_match_trasR.nonzero()).T
        self.A2B_bi_Bs = {u: set(row) for u, row in enumerate(A2B_match_trasR.rows)}

        self.result_queue = Queue(maxsize=n_workers*2)
        self.threads = []
        for i in range(n_workers):
            self.threads.append(Thread(target=self.__sample_function__))
            self.threads[-1].start()

    def __sample_function__(self,):
        while True:
            np.random.shuffle(self.As)
            for batch in range(int(len(self.As) / self.batch_size)):
                users = self.As[batch * self.batch_size: (batch + 1) * self.batch_size]
                uijkl_batch = []
                for u in users:
                    if u in self.A2B_bi_Bs and u in self.A2B_uni_Bs:
                        i, j, k, l = None, None, None, None

                        if len(self.A2B_bi_Bs[u]) != self.n_B and len(self.A2B_bi_Bs[u]) != 0:
                            i = np.random.choice(list(self.A2B_bi_Bs[u]), 1)[0]
                            j = np.random.randint(0, self.n_B, 1)[0]
                            while j in self.A2B_bi_Bs[u]:
                                j = np.random.randint(0, self.n_B, 1)[0]

                        if len(self.A2B_uni_Bs[u]) != self.n_B and len(self.A2B_uni_Bs[u]) != 0:
                            k = np.random.choice(list(self.A2B_uni_Bs[u]), 1)[0]
                            l = np.random.randint(0, self.n_B, 1)[0]
                            while l in self.A2B_uni_Bs[u]:
                                l = np.random.randint(0, self.n_B, 1)[0]
                        if i == j == k == l == None:
                            continue
                        elif i == j == None:
                            i, j = k, l
                        elif k == l == None:
                            k, l = i, j
                        uijkl_batch.append(np.array([u,i,j,k,l]))
                self.result_queue.put(np.asarray(uijkl_batch))

    def next_batch(self):
        return self.result_queue.get()
