import numpy as np
import tensorflow as tf
from tensorflow.python.client.device_lib import list_local_devices
# from tqdm import tqdm

import gc

import sys
sys.path.append("../../metrics/")
from ranking import evaluateCV, evaluateLOOV
sys.path.append("../../samplers/")
from sampler_uijkl_ranking import Sampler

class RRK(object):
    def __init__(self, n_A, n_B, topN=5,
                 split_method='cv', eval_metrics=['pre','recall','mrr', 'ndcg'],
                 reg=0.02, n_factors=128, batch_size=100,
                 max_iter=50, lr=0.1,
                 init_mean=0.0, init_stddev=0.1,
                 device='CPU'):
        # parameters
        self.__n_A, self.__n_B, self.__topN = n_A, n_B, topN
        self.__split_method, self.__eval_metrics = split_method, eval_metrics
        self.__reg, self.__n_factors, self.__batch_size = reg, n_factors, batch_size
        self.__max_iter, self.__lr = max_iter, lr
        self.__init_mean, self.__init_stddev = init_mean, init_stddev

        self.__device = device
        self.__DEVICES = [x.name for x in list_local_devices() if x.device_type == device]

        self.__A_embed = tf.get_variable(shape=[self.__n_A, self.__n_factors],
                                initializer=tf.truncated_normal_initializer(mean=self.__init_mean,stddev=self.__init_stddev),
                                            name='A_embed')
        self.__B_embed = tf.get_variable(shape=[self.__n_B,self.__n_factors],
                                initializer=tf.truncated_normal_initializer(mean=self.__init_mean,stddev=self.__init_stddev),
                                            name='B_embed')
        # data input:
        # A2B
        self.__A2B_uijkl_placeholder = tf.placeholder(tf.int32, shape=[None, 5])
        # evaluate
        self.__A_placeholder = tf.placeholder(tf.int32, shape=[None])

        # B2A
        self.__B2A_uijkl_placeholder = tf.placeholder(tf.int32, shape=[None, 5])
        # evaluate
        self.__B_placeholder = tf.placeholder(tf.int32, shape=[None])
        # property
        self.__A2B_reg_loss__
        self.__B2A_reg_loss__
        self.__A2B_embed_loss__
        self.__B2A_embed_loss__
        self.__loss__
        self.__A2B_predict__
        self.__B2A_predict__
        self.__optimize__

        self.__sess = None
        self.__sampler = None

    @property
    def __A2B_reg_loss__(self):
        reg_loss_ = tf.nn.l2_loss(tf.nn.embedding_lookup(self.__A_embed, self.__A2B_uijkl_placeholder[:, 0]))
        reg_loss_ += tf.nn.l2_loss(tf.nn.embedding_lookup(self.__B_embed, self.__A2B_uijkl_placeholder[:, 1:]))
        return self.__reg * reg_loss_

    @property
    def __B2A_reg_loss__(self):
        reg_loss_ = tf.nn.l2_loss(tf.nn.embedding_lookup(self.__B_embed, self.__B2A_uijkl_placeholder[:, 0]))
        reg_loss_ += tf.nn.l2_loss(tf.nn.embedding_lookup(self.__A_embed, self.__B2A_uijkl_placeholder[:, 1:]))
        return self.__reg * reg_loss_

    @property
    def __A2B_embed_loss__(self):
        # u embedding (N, K)
        A2B_u_embed = tf.nn.embedding_lookup(self.__A_embed, self.__A2B_uijkl_placeholder[:, 0], name="A2B_u_embed_")
        # i item embedding (N, K)
        A2B_i_embed = tf.nn.embedding_lookup(self.__B_embed, self.__A2B_uijkl_placeholder[:, 1], name="A2B_i_embed_")
        # j item embedding (N, K)
        A2B_j_embed = tf.nn.embedding_lookup(self.__B_embed, self.__A2B_uijkl_placeholder[:, 2], name="A2B_j_embed_")
        # k item embedding (N, K)
        A2B_k_embed = tf.nn.embedding_lookup(self.__B_embed, self.__A2B_uijkl_placeholder[:, 3], name="A2B_k_embed_")
        # l item embedding (N, K)
        A2B_l_embed = tf.nn.embedding_lookup(self.__B_embed, self.__A2B_uijkl_placeholder[:, 4], name="A2B_l_embed_")

        A2B_ui = tf.reduce_sum(A2B_u_embed * A2B_i_embed, reduction_indices=1, name="A2B_ui_")
        A2B_uj = tf.reduce_sum(A2B_u_embed * A2B_j_embed, reduction_indices=1, name="A2B_uj_")
        A2B_uk = tf.reduce_sum(A2B_u_embed * A2B_k_embed, reduction_indices=1, name="A2B_uk_")
        A2B_ul = tf.reduce_sum(A2B_u_embed * A2B_l_embed, reduction_indices=1, name="A2B_ul_")

        A2B_uij = tf.reduce_sum(-tf.log(tf.sigmoid(A2B_ui - A2B_uj)), name="loss_A2B_uij_")
        A2B_ukl = tf.reduce_sum(-tf.log(tf.sigmoid(A2B_uk - A2B_ul)), name="loss_A2B_ukl_")
        return A2B_uij + A2B_ukl

    @property
    def __B2A_embed_loss__(self):
        # u embedding (N, K)
        B2A_u_embed = tf.nn.embedding_lookup(self.__B_embed, self.__B2A_uijkl_placeholder[:, 0], name="B2A_u_embed_")
        # i item embedding (N, K)
        B2A_i_embed = tf.nn.embedding_lookup(self.__A_embed, self.__B2A_uijkl_placeholder[:, 1], name="B2A_i_embed_")
        # j item embedding (N, K)
        B2A_j_embed = tf.nn.embedding_lookup(self.__A_embed, self.__B2A_uijkl_placeholder[:, 2], name="B2A_j_embed_")
        # k item embedding (N, K)
        B2A_k_embed = tf.nn.embedding_lookup(self.__A_embed, self.__B2A_uijkl_placeholder[:, 3], name="B2A_k_embed_")
        # l item embedding (N, K)
        B2A_l_embed = tf.nn.embedding_lookup(self.__A_embed, self.__B2A_uijkl_placeholder[:, 4], name="B2A_l_embed_")

        B2A_ui = tf.reduce_sum(B2A_u_embed * B2A_i_embed, reduction_indices=1, name="B2A_ui_")
        B2A_uj = tf.reduce_sum(B2A_u_embed * B2A_j_embed, reduction_indices=1, name="B2A_uj_")
        B2A_uk = tf.reduce_sum(B2A_u_embed * B2A_k_embed, reduction_indices=1, name="B2A_uk_")
        B2A_ul = tf.reduce_sum(B2A_u_embed * B2A_l_embed, reduction_indices=1, name="B2A_ul_")

        B2A_uij = tf.reduce_sum(-tf.log(tf.sigmoid(B2A_ui - B2A_uj)), name="loss_B2A_uij_")
        B2A_ukl = tf.reduce_sum(-tf.log(tf.sigmoid(B2A_uk - B2A_ul)), name="loss_B2A_ukl_")
        return B2A_uij + B2A_ukl

    @property
    def __loss__(self):
        return self.__A2B_embed_loss__ + self.__A2B_reg_loss__ + self.__B2A_reg_loss__ + self.__B2A_embed_loss__

    @property
    def __A2B_predict__(self):
        tst_A_embed = tf.nn.embedding_lookup(self.__A_embed, self.__A_placeholder, name="tst_A_embed_")
        A2B_predicts = tf.matmul(tst_A_embed, tf.transpose(self.__B_embed), name='A2B_predicts_')
        return A2B_predicts

    @property
    def __B2A_predict__(self):
        tst_B_embed = tf.nn.embedding_lookup(self.__B_embed, self.__B_placeholder, name="tst_B_embed_")
        B2A_predicts = tf.matmul(tst_B_embed, tf.transpose(self.__A_embed), name='B2A_predicts_')
        return B2A_predicts

    @property
    def __optimize__(self):
        gds = []
        gds.append(tf.train.AdagradOptimizer(self.__lr).minimize(self.__loss__,
                    var_list=[self.__A_embed, self.__B_embed]))
        with tf.control_dependencies(gds):
            return gds + [self.__A_embed, self.__B_embed]


    def __A2B_recommend(self, test_As, A2B_tstintra_set):
        itemset_maxsize = max([len(itemset) for itemset in A2B_tstintra_set])

        yss_pred_ = self.__sess.run(tf.nn.top_k(self.__A2B_predict__, itemset_maxsize + self.__topN),
                                    {self.__A_placeholder: test_As})[1]
        # filter out the rated items
        yss_pred = []
        for ind in range(len(test_As)):
            yss_pred.append([])
            for y_pred_ in yss_pred_[ind]:
                if y_pred_ not in A2B_tstintra_set[ind]:
                    yss_pred[-1].append(y_pred_)
                if len(yss_pred[-1]) >= self.__topN:
                    break
        return yss_pred

    def __B2A_recommend(self, test_Bs, B2A_tstintra_set):
        itemset_maxsize = max([len(itemset) for itemset in B2A_tstintra_set])

        yss_pred_ = self.__sess.run(tf.nn.top_k(self.__B2A_predict__, itemset_maxsize + self.__topN),
                                    {self.__B_placeholder: test_Bs})[1]
        # filter out the rated items
        yss_pred = []
        for ind in range(len(test_Bs)):
            yss_pred.append([])
            for y_pred_ in yss_pred_[ind]:
                if y_pred_ not in B2A_tstintra_set[ind]:
                    yss_pred[-1].append(y_pred_)
                if len(yss_pred[-1]) >= self.__topN:
                    break
        return yss_pred

    def __eval(self, yss_true, yss_pred):
        if self.__split_method=='cv':
            return evaluateCV(yss_true, yss_pred, self.__eval_metrics, self.__topN)
        elif self.__split_method=='loov':
            return evaluateLOOV(yss_true, yss_pred, self.__eval_metrics, self.__topN)
        else:
            return None

    def __getUniTst(self, A2B_tstsR):
        uni_tst_users = list(set(np.asarray(A2B_tstsR.nonzero()[0])))
        uni_yss_true = None
        if self.__split_method == 'cv':
            uni_yss_true = [set(A2B_tstsR[user].nonzero()[1]) for user in uni_tst_users]
        elif self.__split_method == 'loov':
            uni_yss_true = [A2B_tstsR[user].nonzero()[1][0] for user in uni_tst_users]
        return uni_tst_users, uni_yss_true

    def __getBiTst(self, A2B_tstsR, B2A_tstsR):
        A2B_match_tstsR = A2B_tstsR.multiply(B2A_tstsR.T)>0
        # match_tst
        bi_tst_users = list(set(np.asarray(A2B_match_tstsR.nonzero()[0])))
        bi_yss_true = None
        if self.__split_method == 'cv':
            bi_yss_true = [set(A2B_match_tstsR[user].nonzero()[1]) for user in bi_tst_users]
        elif self.__split_method == 'loov':
            bi_yss_true = [A2B_match_tstsR[user].nonzero()[1][0] for user in bi_tst_users]
        return bi_tst_users, bi_yss_true

    def train(self, fold, A2B_trasR, A2B_tstsR, B2A_trasR, B2A_tstsR):
        __A2B_sampler = Sampler(A2B_trasR, B2A_trasR, self.__batch_size)
        __B2A_sampler = Sampler(B2A_trasR, A2B_trasR, self.__batch_size)

        # for eval:
        # A2B_tst
        A2B_uni_tst_users, A2B_uni_yss_true = self.__getUniTst(A2B_tstsR)
        A2B_bi_tst_users, A2B_bi_yss_true = self.__getBiTst(A2B_tstsR, B2A_tstsR)
        # tra
        A2B_uni_tstintra_set = [set(A2B_trasR[user].nonzero()[1]) for user in A2B_uni_tst_users]
        A2B_bi_tstintra_set = [set(A2B_trasR[user].nonzero()[1]) for user in A2B_bi_tst_users]

        # B2A_tst
        B2A_uni_tst_users, B2A_uni_yss_true = self.__getUniTst(B2A_tstsR)
        B2A_bi_tst_users, B2A_bi_yss_true = self.__getBiTst(B2A_tstsR, A2B_tstsR)
        # tra
        B2A_uni_tstintra_set = [set(B2A_trasR[user].nonzero()[1]) for user in B2A_uni_tst_users]
        B2A_bi_tstintra_set = [set(B2A_trasR[user].nonzero()[1]) for user in B2A_bi_tst_users]

        # for train:
        A2B_tra_tuple = np.array([(user, item) for user, item in np.asarray(A2B_trasR.nonzero()).T])  # double
        B2A_tra_tuple = np.array([(user, item) for user, item in np.asarray(B2A_trasR.nonzero()).T])  # double
        scores = None
        with tf.device(self.__DEVICES[0]):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            train_op = (self.__optimize__, self.__loss__) # must before the initializer
            self.__sess = tf.Session(config=config)
            self.__sess.run(tf.global_variables_initializer())

            n_batches = int(0.5 * (len(A2B_tra_tuple) + len(B2A_tra_tuple))/self.__batch_size)
            # sample all users
            preAveLoss = 0
            for iter in range(self.__max_iter):
                losses = []
                for _ in range(n_batches):
                    A2B_uijkl_batch = __A2B_sampler.next_batch()
                    B2A_uijkl_batch = __B2A_sampler.next_batch()
                    _, loss = self.__sess.run(train_op, {self.__A2B_uijkl_placeholder: A2B_uijkl_batch,
                                                         self.__B2A_uijkl_placeholder: B2A_uijkl_batch})
                    losses.append(loss)
                aveloss = np.mean(losses)

                A2B_uni_yss_pred = self.__A2B_recommend(A2B_uni_tst_users, A2B_uni_tstintra_set)
                B2A_uni_yss_pred = self.__B2A_recommend(B2A_uni_tst_users, B2A_uni_tstintra_set)

                uni_scores = self.__eval(A2B_uni_yss_true + B2A_uni_yss_true, A2B_uni_yss_pred + B2A_uni_yss_pred)

                A2B_bi_yss_pred = self.__A2B_recommend(A2B_bi_tst_users, A2B_bi_tstintra_set)
                B2A_bi_yss_pred = self.__B2A_recommend(B2A_bi_tst_users, B2A_bi_tstintra_set)
                bi_scores = self.__eval(A2B_bi_yss_true + B2A_bi_yss_true, A2B_bi_yss_pred + B2A_bi_yss_pred)

                print("%s_fold=%d iter=%2d: " % (self.__split_method, fold, iter + 1),
                      "TraLoss=%.4f delta_loss=%.4f lr=%.4f" % (aveloss, aveloss-preAveLoss, self.__lr),' ',
                      ','.join(['%s' % eval_metric for eval_metric in self.__eval_metrics]) + '@%d:' % (self.__topN))
                # unilateral and bilateral
                print('\tTst: uni=' + ','.join(['%.6f' % (score) for score in uni_scores])
                    +'  bi=' +','.join(['%.6f' % (score) for score in bi_scores]))

                scores = [uni_scores, bi_scores]
                preAveLoss = aveloss
                self.__lr *= .98
                gc.collect()
        return scores

    def close(self):
        self.__sess.close()
