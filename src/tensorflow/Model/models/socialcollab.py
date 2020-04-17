
import numpy as np

import gc

import sys
sys.path.append("../../metrics/")
from ranking import evaluateCV, evaluateLOOV


class SocialCollab(object):
    def __init__(self,
                 topN=5, split_method='cv', eval_metrics=['rmse', 'mae']):
        self.__topN, self.__split_method, self.__eval_metrics = topN, split_method, eval_metrics

    def __tasteMat__(self, sR): # A2B
        tasteSimMat = (sR * sR.T)>0
        tasteMat = tasteSimMat * sR
        return tasteMat

    def __attractMat__(self, sR): # B2A
        attractSimMat = (sR.T * sR)>0
        attractMat = attractSimMat * sR.T
        return attractMat

    def __predict(self, A2B_trasR, B2A_trasR):
        A2B_tasteMat, A2B_attractMat = self.__tasteMat__(A2B_trasR), self.__attractMat__(B2A_trasR)
        A2B_boolMat = A2B_tasteMat.multiply(A2B_attractMat) > 0
        A2B_predict = A2B_tasteMat.multiply(A2B_boolMat) + A2B_attractMat.multiply(A2B_boolMat) #
        return A2B_predict

    def __eval(self, yss_true, yss_pred):
        if self.__split_method=='cv':
            return evaluateCV(yss_true, yss_pred, self.__eval_metrics, self.__topN)
        elif self.__split_method=='loov':
            return evaluateLOOV(yss_true, yss_pred, self.__eval_metrics, self.__topN)
        else:
            return None

    def __recommend(self, A2B_trasR, A2B_predict, test_As):
        itemset_maxsize = max([A2B_trasR[rid,:].nnz for rid in range(A2B_trasR.shape[0])])

        yss_pred = []
        # print(A2B_predict.nnz, len(set(A2B_predict.nonzero()[0])))
        for a in test_As:
            yss_pred.append([])
            for item in np.argsort(A2B_predict[a,:].toarray()[0])[-itemset_maxsize-self.__topN:][::-1]:
                if A2B_trasR[a, item]==0:
                    yss_pred[-1].append(item)
                if len(yss_pred[-1]) >= self.__topN:
                    break
        return yss_pred

    def train(self, fold, A2B_trasR, B2A_trasR, A2B_match_tstsR):
        # for eval:
        # tst
        tst_As = list(set(A2B_match_tstsR.nonzero()[0]))
        tst_Bs = list(set(A2B_match_tstsR.nonzero()[1]))

        # A2B
        A2B_predict = self.__predict(A2B_trasR, B2A_trasR)
        A2B_rec = self.__recommend(A2B_trasR, A2B_predict, tst_As)
        # B2A
        B2A_predict = self.__predict(B2A_trasR, A2B_trasR)
        B2A_rec = self.__recommend(B2A_trasR, B2A_predict, tst_Bs)

        A2B_true, B2A_true = None, None
        if self.__split_method=='cv':
            A2B_true = [set(A2B_match_tstsR[user,:].nonzero()[1]) for user in tst_As]
            B2A_true = [set(A2B_match_tstsR.T[user,:].nonzero()[1]) for user in tst_Bs]
        elif self.__split_method=='loov':
            A2B_true = [A2B_match_tstsR[user].nonzero()[1][0] for user in tst_As]
            B2A_true = [A2B_match_tstsR.T[user].nonzero()[1][0] for user in tst_Bs]

        A2B_scores = self.__eval(A2B_true, A2B_rec)
        B2A_scores = self.__eval(B2A_true, B2A_rec)
        return A2B_scores, B2A_scores





