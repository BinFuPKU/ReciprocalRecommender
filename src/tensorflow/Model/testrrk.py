
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("./models/")
from rrk import RRK
sys.path.append("../../utils/")
from IOUtil import loadSparseR

import multiprocessing
from scipy.sparse import lil_matrix
import numpy as np

data_dir = '../../../data/'
folds = 5

reg = .001

topN = 20
split_method='cv'
eval_metrics = ['pre','recall', 'map', 'mrr', 'ndcg']
n_factors=100
batch_size=100

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

    rrtcf = RRK(n_A, n_B, topN,
                 split_method, eval_metrics,
                 reg, n_factors, batch_size)
    uni_scores, bi_scores = rrtcf.train(fold+1, A2B_trasR, A2B_tstsR, B2A_trasR, B2A_tstsR)

    # print results
    print(','.join(['%s' % eval_metric for eval_metric in eval_metrics]) + '@%d:' % (topN))
    # unilateral and bilateral
    print(dataset_dir.split('/')[-2] + '@%d: uni@%d=' % (fold + 1, topN) + ','.join(['%.6f' % (score) for score in uni_scores])
          +'  bi@%d=' % (topN) +','.join(['%.6f' % (score) for score in bi_scores]))

    return [uni_scores, bi_scores]

def avestd_(scores, name):
    aves = scores.sum(0) / len(scores)
    stds = np.sqrt(np.power(np.array(scores) - aves, 2).sum(0) / len(scores))
    print(name+':', 'ave@' + str(topN) + '=[' + ','.join(['%.4f' % ave for ave in aves]) + ']',
          'std@' + str(topN) + '=[' + ','.join(['%.4f' % std for std in stds]) + ']')

if __name__ =='__main__':
    print('reg=', reg)
    # kanzhun
    # dataset_dir = data_dir+'/kanzhun/'
    # n_A, n_B = 799, 993
    # A_name, B_name = 'E', 'J'

    # libimseti
    dataset_dir = data_dir+'/libimseti/'
    n_A, n_B = 3005, 3027
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

    # 5 * 2 * 5
    scores = np.array([result.get() for result in results])
    avestd_(scores[:,0,:], 'unilateral')
    avestd_(scores[:,1,:], 'bilateral')
