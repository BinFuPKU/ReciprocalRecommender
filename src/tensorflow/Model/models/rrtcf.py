import numpy as np
import tensorflow as tf
from tensorflow.python.client.device_lib import list_local_devices
# from tqdm import tqdm

import gc
import datetime as dt
from copy import deepcopy as dc
from src.eval.evaluate import Evaluator
from src.samplers.sampler_bi_ranking import Sampler

class RRTCF(object):
    def __init__(self, n_A, n_B, topNs=[10, 20, 50, 100],
                 alpha=0.5, beta=0.1, lambda_=0.1,
                 split_method='cv', eval_metrics=['pre','recall','mrr', 'ndcg'],
                 n_factors=20, batch_size=100,
                 max_iter=30, lr=0.02,
                 init_mean=0.0, init_stddev=0.1,
                 device='CPU'):
        # parameters
        self.n_A, self.n_B, self.topNs = n_A, n_B, topNs
        self.alpha, self.beta, self.lambda_ = alpha, beta, lambda_
        self.n_factors, self.batch_size = n_factors, batch_size
        self.max_iter, self.lr = max_iter, lr

        self.split_method = split_method
        self.evaluator = Evaluator(split_method, eval_metrics, topNs)

        self.device = device
        self.DEVICES = [x.name for x in list_local_devices() if x.device_type == device]

        self.A_embed = tf.get_variable(shape=[self.n_A, self.n_factors],
                                initializer=tf.truncated_normal_initializer(mean=init_mean,stddev=init_stddev),
                                            name='A_embed')
        self.A2B_kernel = tf.get_variable(shape=[self.n_factors,self.n_factors],
                                initializer=tf.truncated_normal_initializer(mean=init_mean,stddev=init_stddev),
                                            name='A2B_kernel')
        self.B2A_kernel = tf.get_variable(shape=[self.n_factors,self.n_factors],
                                initializer=tf.truncated_normal_initializer(mean=init_mean,stddev=init_stddev),
                                            name='B2A_kernel')
        self.B_embed = tf.get_variable(shape=[self.n_B,self.n_factors],
                                initializer=tf.truncated_normal_initializer(mean=init_mean,stddev=init_stddev),
                                            name='B_embed')

        # data input:
        self.A2B_useritem_placeholder = tf.placeholder(tf.int32, shape=[None, 2])
        self.A2B_negItems_placeholder = tf.placeholder(tf.int32, shape=[None, None])
        self.B2A_useritem_placeholder = tf.placeholder(tf.int32, shape=[None, 2])
        self.B2A_negItems_placeholder = tf.placeholder(tf.int32, shape=[None, None])

        # evaluate
        self.A_users_placeholder = tf.placeholder(tf.int32, shape=[None])
        self.B_users_placeholder = tf.placeholder(tf.int32, shape=[None])

        # property
        self.A2B_reg_loss
        self.B2A_reg_loss
        self.A2B_embed_loss
        self.B2A_embed_loss
        self.loss
        self.A2B_predict
        self.B2A_predict
        self.optimize

        self.sess = None
        self.sampler = None

    @property
    def A2B_reg_loss(self):
        reg_loss_ = self.alpha * tf.nn.l2_loss(tf.nn.embedding_lookup(self.A_embed, self.A2B_useritem_placeholder[:, 0])) + \
                    self.alpha * tf.nn.l2_loss(tf.nn.embedding_lookup(self.B_embed, self.A2B_useritem_placeholder[:, 1])) + \
                    self.alpha * tf.nn.l2_loss(tf.nn.embedding_lookup(self.B_embed, self.A2B_negItems_placeholder)) + \
                    self.beta * tf.nn.l2_loss(self.A2B_kernel)
        return reg_loss_

    @property
    def B2A_reg_loss(self):
        reg_loss_ = self.alpha * tf.nn.l2_loss(tf.nn.embedding_lookup(self.B_embed, self.B2A_useritem_placeholder[:, 0])) + \
                    self.alpha * tf.nn.l2_loss(tf.nn.embedding_lookup(self.A_embed, self.B2A_useritem_placeholder[:, 1])) + \
                    self.alpha * tf.nn.l2_loss(tf.nn.embedding_lookup(self.A_embed, self.B2A_negItems_placeholder)) + \
                    self.beta * tf.nn.l2_loss(self.B2A_kernel)
        return reg_loss_

    @property
    def A2B_embed_loss(self):
        # user embedding (N, K)
        user_embed = tf.nn.embedding_lookup(self.A_embed, self.A2B_useritem_placeholder[:, 0], name="A2B_user_embed_")
        # positive item embedding (N, K)
        posItem_embed = tf.nn.embedding_lookup(self.B_embed, self.A2B_useritem_placeholder[:, 1], name="A2B_posItem_embed_")
        # negative item embedding (N, W, K)
        negItems_embed = tf.nn.embedding_lookup(self.B_embed, self.A2B_negItems_placeholder, name="A2B_negItems_embed_")

        # N
        ui = tf.reduce_sum(tf.multiply(tf.matmul(user_embed, self.A2B_kernel), posItem_embed),
                                       reduction_indices=-1, name='A2B_ui_')
        # N * W
        uj = tf.reduce_sum(tf.multiply(tf.expand_dims(tf.matmul(user_embed, self.A2B_kernel), 1), negItems_embed),
                           reduction_indices=-1, name='A2B_uj_')
        embed_loss_ = tf.reduce_sum(-tf.log(tf.sigmoid(tf.expand_dims(ui,-1) - uj)), name="A2B_embed_loss_")
        return embed_loss_

    @property
    def B2A_embed_loss(self):
        # user embedding (N, K)
        user_embed = tf.nn.embedding_lookup(self.B_embed, self.B2A_useritem_placeholder[:, 0], name="B2A_user_embed_")
        # positive item embedding (N, K)
        posItem_embed = tf.nn.embedding_lookup(self.A_embed, self.B2A_useritem_placeholder[:, 1], name="B2A_posItem_embed_")
        # negative item embedding (N, W, K)
        negItems_embed = tf.nn.embedding_lookup(self.A_embed, self.B2A_negItems_placeholder, name="B2A_negItems_embed_")

        # N
        ui = tf.reduce_sum(tf.multiply(tf.matmul(user_embed, self.B2A_kernel), posItem_embed),
                                       reduction_indices=-1, name='B2A_ui_')
        # N * W
        uj = tf.reduce_sum(tf.multiply(tf.expand_dims(tf.matmul(user_embed, self.B2A_kernel), 1), negItems_embed),
                           reduction_indices=-1, name='B2A_uj_')
        embed_loss_ = tf.reduce_sum(-tf.log(tf.sigmoid(tf.expand_dims(ui,-1) - uj)), name="B2A_embed_loss_")
        return embed_loss_

    @property
    def loss(self):
        return self.A2B_embed_loss + self.A2B_reg_loss + \
               self.lambda_ * (self.B2A_embed_loss + self.B2A_reg_loss)

    @property
    def A2B_predict(self):
        user_embed = tf.nn.embedding_lookup(self.A_embed, self.A_users_placeholder, name="A2B_user_embed_predict")
        A2B_rating = tf.matmul(tf.matmul(user_embed, self.A2B_kernel), tf.transpose(self.B_embed), name='A2B_A2B_rating_')
        B2A_rating = tf.matmul(tf.matmul(self.B_embed, self.B2A_kernel), tf.transpose(user_embed), name='A2B_B2A_rating_')
        return tf.multiply(A2B_rating, tf.transpose(B2A_rating))

    @property
    def B2A_predict(self):
        user_embed = tf.nn.embedding_lookup(self.B_embed, self.B_users_placeholder, name="B2A_user_embed_predict")
        B2A_rating = tf.matmul(tf.matmul(user_embed, self.B2A_kernel), tf.transpose(self.A_embed), name='B2A_B2A_rating_')
        A2B_rating = tf.matmul(tf.matmul(self.A_embed, self.A2B_kernel), tf.transpose(user_embed), name='B2A_A2B_rating_')
        return tf.multiply(B2A_rating, tf.transpose(A2B_rating))

    @property
    def optimize(self):
        gds = []
        gds.append(tf.train.AdamOptimizer(self.lr).minimize(self.loss,
                    var_list=[self.A_embed, self.A2B_kernel, self.B_embed, self.B2A_kernel]))
        with tf.control_dependencies(gds):
            return gds + [self.A_embed, self.A2B_kernel, self.B_embed, self.B2A_kernel]


    def A2B_recommend(self, test_As, A2B_tstintra_set, topN):
        itemset_maxsize = max([len(itemset) for itemset in A2B_tstintra_set])

        yss_pred_ = self.sess.run(tf.nn.top_k(self.A2B_predict, itemset_maxsize + topN),
                                    {self.A_users_placeholder: test_As})[1]
        # filter out the rated items
        yss_pred = []
        for ind in range(len(test_As)):
            yss_pred.append([])
            for y_pred_ in yss_pred_[ind]:
                if y_pred_ not in A2B_tstintra_set[ind]:
                    yss_pred[-1].append(y_pred_)
                if len(yss_pred[-1]) >= topN:
                    break
        return yss_pred

    def B2A_recommend(self, test_Bs, B2A_tstintra_set, topN):
        itemset_maxsize = max([len(itemset) for itemset in B2A_tstintra_set])

        yss_pred_ = self.sess.run(tf.nn.top_k(self.B2A_predict, itemset_maxsize + topN),
                                    {self.B_users_placeholder: test_Bs})[1]
        # filter out the rated items
        yss_pred = []
        for ind in range(len(test_Bs)):
            yss_pred.append([])
            for y_pred_ in yss_pred_[ind]:
                if y_pred_ not in B2A_tstintra_set[ind]:
                    yss_pred[-1].append(y_pred_)
                if len(yss_pred[-1]) >= topN:
                    break
        return yss_pred

    def train(self, fold, A2B_trasR, A2B_tstsR, B2A_trasR, B2A_tstsR, negSample):
        # sampler
        self.sampler = Sampler(A2B_trasR, B2A_trasR, negSample, self.batch_size)

        # test
        # match_tst
        match_A2B_tstsR = dc(A2B_tstsR).multiply(B2A_tstsR.T)>0
        A_tst_users, B_tst_users = list(set(np.asarray(match_A2B_tstsR.nonzero()[0]))),\
                                   list(set(np.asarray(match_A2B_tstsR.nonzero()[1])))
        A_yss_true, B_yss_true = None, None
        if self.split_method=='cv':
            A_yss_true = [set(match_A2B_tstsR[user, :].nonzero()[1]) for user in A_tst_users]
            B_yss_true = [set(match_A2B_tstsR[:, user].nonzero()[0]) for user in B_tst_users]
        elif self.split_method=='loov':
            A_yss_true = [match_A2B_tstsR[user, :].nonzero()[1][0] for user in A_tst_users]
            B_yss_true = [match_A2B_tstsR[:, user].nonzero()[0][0] for user in B_tst_users]

        # tra
        A2B_trasR_ = dc(A2B_trasR) + B2A_trasR.T
        A2B_tstintra_set = [set(A2B_trasR_[user, :].nonzero()[1]) for user in A_tst_users]
        B2A_tstintra_set = [set(A2B_trasR_[:, user].nonzero()[0]) for user in B_tst_users]


        scoress = None
        with tf.device(self.DEVICES[0]):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            train_op = (self.optimize, self.loss) # must before the initializer
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

            n_batches = int(A2B_trasR.nnz/self.batch_size)

            # sample all users
            preAveLoss = 0
            for iter in range(self.max_iter):
                losses = []
                for _ in range(n_batches):
                    A_batch, B_batch = self.sampler.next_batch()

                    __, loss = self.sess.run(train_op, {self.A2B_useritem_placeholder: A_batch[:, :2],
                                                        self.A2B_negItems_placeholder: A_batch[:, 2:],
                                                        self.B2A_useritem_placeholder: B_batch[:, :2],
                                                        self.B2A_negItems_placeholder: B_batch[:, 2:]})

                    losses.append(loss)
                aveloss = np.mean(losses)

                A_yss_pred = self.A2B_recommend(A_tst_users, A2B_tstintra_set, max(self.evaluator.topNs))
                A_scoress = self.evaluator.eval(A_yss_true, A_yss_pred)

                B_yss_pred = self.B2A_recommend(B_tst_users, B2A_tstintra_set, max(self.evaluator.topNs))
                B_scoress = self.evaluator.eval(B_yss_true, B_yss_pred)

                scoress = []
                for i in range(len(self.topNs)):
                    scores = []
                    for j in range(len(self.evaluator.eval_metrics)):
                        score = (A_scoress[i][j] * len(A_tst_users) + B_scoress[i][j] * len(B_tst_users))\
                                / (len(A_tst_users) + len(B_tst_users))
                        scores.append(score)
                    scoress.append(scores)

                print(dt.datetime.now().strftime('%m-%d %H:%M:%S'),
                      "%s_fold=%d iter=%2d: " % (self.split_method, fold, iter + 1),
                      "TraLoss=%.4f delta_loss=%.4f lr=%.4f" % (aveloss, aveloss-preAveLoss, self.lr))
                print('\tTst:',
                      ','.join(['%s' % eval_metric for eval_metric in self.evaluator.eval_metrics]) +
                      '@%d=' % (self.evaluator.topNs[1]),
                      ','.join(['%.6f' % (score) for score in scoress[1]]))

                preAveLoss = aveloss
                # self.lr *= .95
                # gc.collect()
        return scoress

    def close(self):
        self.sess.close()
