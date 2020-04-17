
import sys
sys.path.append("./models/")
from socialcollab import SocialCollab
sys.path.append("../../utils/")
from IOUtil import loadSparseR

import multiprocessing
from scipy.sparse import lil_matrix
import numpy as np

data_dir = '../../../data/'
folds = 5

topN = 100
split_method='cv'
eval_metrics = ['pre','recall', 'map', 'mrr', 'ndcg']

def worker(fold, n_A, n_B, dataset_dir, A_name, B_name):
    A2B_trasR = lil_matrix(loadSparseR(n_A, n_B, dataset_dir+A_name+'2'+B_name+'_'+str(fold + 1) + '_tra.txt'))
    B2A_trasR = lil_matrix(loadSparseR(n_B, n_A, dataset_dir+B_name+'2'+A_name+'_'+str(fold + 1) + '_tra.txt'))

    trasR = lil_matrix((n_A + n_B, n_A + n_B))
    trasR[:n_A, :n_B] = A2B_trasR
    trasR[n_A:, n_B:] = B2A_trasR
    print(dataset_dir.split('/')[-2] + '@%d:' % (fold + 1), trasR.shape, trasR.nnz,
          '%.2f' % (trasR.nnz / float(trasR.shape[0])))

    A2B_tstsR = lil_matrix(loadSparseR(n_A, n_B, dataset_dir+A_name+'2'+B_name+'_'+str(fold + 1) + '_tst.txt'))
    B2A_tstsR = lil_matrix(loadSparseR(n_B, n_A, dataset_dir+B_name+'2'+A_name+'_'+str(fold + 1) + '_tst.txt'))

    A2B_match_tstsR = A2B_tstsR.multiply(B2A_tstsR.T)>0

    sc = SocialCollab(topN, split_method, eval_metrics)
    A2B_scores, B2A_scores = sc.train(fold+1, A2B_trasR, B2A_trasR, A2B_match_tstsR)

    # bilateral_scores
    tst_Anum, tst_Bnum = len(set(A2B_match_tstsR.nonzero()[0])), len(set(A2B_match_tstsR.nonzero()[1]))
    bi_scores = [
        tst_Anum / float(tst_Anum + tst_Bnum) * A2B_scores[i] +
        tst_Bnum / float(tst_Anum + tst_Bnum) * B2A_scores[i]
        for i in range(len(eval_metrics))]

    # print results
    print(','.join(['%s' % eval_metric for eval_metric in eval_metrics]) + '@%d:' % (topN))
    # bilateral
    print(dataset_dir.split('/')[-2] + '@%d: bi@%d=' % (fold + 1, topN) + ','.join(['%.6f' % (score) for score in bi_scores]))
    return bi_scores

def avestd_(scores, name):
    aves = scores.sum(0) / len(scores)
    stds = np.sqrt(np.power(np.array(scores) - aves, 2).sum(0) / len(scores))
    print(name+':', 'ave@' + str(topN) + '=[' + ','.join(['%.4f' % ave for ave in aves]) + ']',
          'std@' + str(topN) + '=[' + ','.join(['%.4f' % std for std in stds]) + ']')

if __name__ =='__main__':
    # kanzhun
    # dataset_dir = data_dir+'/kanzhun/'
    # n_A, n_B = 799, 993
    # A_name, B_name = 'E', 'J'

    # libimseti
    dataset_dir = data_dir+'/libimseti/'
    n_A, n_B = 3266, 3370
    A_name, B_name = 'F', 'M'

    # process pool
    folds_ = 1
    pool = multiprocessing.Pool(processes=folds_)
    results = []
    for fold in range(folds_):
        results.append(pool.apply_async(func=worker, args=(fold, n_A, n_B, dataset_dir, A_name, B_name)))
    pool.close()
    pool.join()
    pool.terminate()

    # 5 * 5
    # scores = np.array([result.get() for result in results])
    # avestd_(scores, 'bilateral')
