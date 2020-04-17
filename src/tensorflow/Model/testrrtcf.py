
import warnings
warnings.filterwarnings("ignore")

from src.Model.RRS.models.rrtcf import RRTCF
from src.utils.IOUtil import loadSparseR
from src.utils.Util import avestd

import multiprocessing
from scipy.sparse import lil_matrix
import numpy as np

data_dir = '../../../data/'
folds = 5

alpha = .5
beta = 5
lambda_ = 1.5

topNs = [10, 20, 50, 100]
split_method='cv'
eval_metrics = ['pre', 'recall', 'map', 'mrr', 'ndcg']
n_factors=20
batch_size=500
negSample=5

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

    rrtcf = RRTCF(n_A, n_B, topNs, alpha, beta, lambda_, split_method, eval_metrics, n_factors, batch_size)
    scoress = rrtcf.train(fold+1, A2B_trasR, A2B_tstsR, B2A_trasR, B2A_tstsR, negSample)

    return scoress

if __name__ =='__main__':

    # just for test: kanzhun
    # dataset_dir = data_dir+'kanzhun/1-split/'
    # n_A, n_B = 799, 993
    # A_name, B_name = 'E', 'J'

    # libimseti
    # dataset_dir = data_dir+'libimseti/1-split/'
    # n_A, n_B = 4220, 4384
    # A_name, B_name = 'F', 'M'

    # bosszp_beijing_jinrong
    # dataset_dir = data_dir+'bosszp_beijing_jinrong/1-split/'
    # n_A, n_B = 921, 1120
    # A_name, B_name = 'E', 'J'

    # bosszp_chengdu_jishu
    dataset_dir = data_dir+'bosszp_chengdu_jishu/1-split/'
    n_A, n_B = 18372, 16404
    A_name, B_name = 'E', 'J'

    # bosszp_beijing_jishu
    # dataset_dir = data_dir+'bosszp_beijing_jishu/1-split/'
    # n_A, n_B = 20370, 11923
    # A_name, B_name = 'E', 'J'

    print(dataset_dir.split('/')[-3]+':', 'alpha=', alpha, 'beta=', beta, 'lambda_=',lambda_)

    # process pool
    folds_ = 1
    pool = multiprocessing.Pool(processes=folds_)
    results = []
    for fold in range(folds_):
        results.append(pool.apply_async(func=worker, args=(fold, n_A, n_B, dataset_dir, A_name, B_name)))
    # pool.close()
    # pool.join()
    # pool.terminate()

    # |folds| * |topNs| * |eval_metrics|
    scoresss = np.array([result.get() for result in results])
    avestd(eval_metrics, scoresss, topNs)
