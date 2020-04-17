import numpy as np

'''
for more information, please see https://www.jianshu.com/p/7bc96f9f1611.
'''

'''
for many users, and many items (as a set) for each user.
like k-fold-cross-validation.
'''
def precision_k_score(yss_true, yss_pred, k=5):
    if len(yss_true) != len(yss_pred) or len(yss_true)==0 or k<=0:
        raise ValueError('len(yss_true) != len(yss_pred) or len(yss_true)==0 or k<=0!')
    score=.0
    for ind in range(len(yss_true)):
        score += len(set(yss_pred[ind][:k]).intersection(yss_true[ind])) / float(k)
    return score/len(yss_true)


def recall_k_score(yss_true, yss_pred, k=5):
    if len(yss_true) != len(yss_pred) or len(yss_true)==0 or k<=0:
        raise ValueError('len(yss_true) != len(yss_pred) or len(yss_true)==0 or k<=0!')
    score=.0
    for ind in range(len(yss_true)):
        score += len(set(yss_pred[ind][:k]).intersection(yss_true[ind])) / max(float(len(yss_true[ind])), 1.0)
    return score/len(yss_true)


def ndcg_k_score(yss_true, yss_pred, k=5):
    if len(yss_true) != len(yss_pred) or len(yss_true)==0 or k<=0:
        raise ValueError('len(yss_true) != len(yss_pred) or len(yss_true)==0 or k<=0!')
    score=.0
    for ind in range(len(yss_true)):
        label_list = [(yss_pred[ind][i] in yss_true[ind])
                            for i in range(len(yss_pred[ind][:k]))]
        actual_dcg = sum([(2 ** label_list[i] - 1.0) / np.log2(i + 2.0) for i in range(len(label_list))])
        ideal_list = sorted(label_list, reverse=True)
        ideal_dcg = sum([(2 ** ideal_list[i] - 1.0) / np.log2(i + 2.0) for i in range(len(ideal_list))])
        score += actual_dcg / max(ideal_dcg, 1.0)
    return score/len(yss_true)

# map
def map_k_score(yss_true, yss_pred, k=5):
    if len(yss_true) != len(yss_pred) or len(yss_true)==0 or k<=0:
        raise ValueError('len(yss_true) != len(yss_pred) or len(yss_true)==0 or k<=0!')
    score = 0
    for ind in range(len(yss_true)):
        rank, score_ = 0, 0
        for i, x in enumerate(yss_pred[ind][:k]):
            if x in yss_true[ind]:
                rank += 1
                score_ += rank / (i + 1.0)
        score += score_ / len(yss_true[ind])
    return score/len(yss_true)

# mrr: the first one appeared in ground truth.
# https://github.com/miyamamoto/nrr/blob/master/nrr.py
def mrr_k_score(yss_true, yss_pred, k=5):
    if len(yss_true) != len(yss_pred) or len(yss_true)==0 or k<=0:
        raise ValueError('len(yss_true) != len(yss_pred) or len(yss_true)==0 or k<=0!')
    score = 0
    for ind in range(len(yss_true)):
        for i, x in enumerate(yss_pred[ind][:k]):
            if x in yss_true[ind]:
                score += 1 / (i + 1.0)
                break
    return score/len(yss_true)


'''
for many users, and one item for each user.
like left one-out validation
'''
# hit rate
def hr_k_score(ys_true, yss_pred, k=5):
    if len(ys_true) != len(yss_pred) or len(ys_true)==0 or k<=0:
        raise ValueError('len(ys_true) != len(yss_pred) or len(ys_true)==0 or k<=0!')
    score=.0
    for ind in range(len(ys_true)):
        score += ys_true[ind] in set(yss_pred[ind][:k])
    return score

# average hit rate
def arhr_k_score(ys_true, yss_pred, k=5):
    if len(ys_true) != len(yss_pred) or len(ys_true)==0 or k<=0:
        raise ValueError('len(ys_true) != len(yss_pred) or len(ys_true)==0 or k<=0!')
    score=.0
    for ind in range(len(ys_true)):
        if ys_true[ind] in yss_pred[ind][:k]:
            score += 1.0 / (yss_pred[ind][:k].index(ys_true[ind])+1)
    return score


def evaluateCV(yss_true, yss_pred, eval_metrics, k=5):
    scores = []
    for eval_metric in eval_metrics:
        if eval_metric=='pre':
            scores.append(precision_k_score(yss_true, yss_pred, k))
        elif eval_metric=='recall':
            scores.append(recall_k_score(yss_true, yss_pred, k))
        elif eval_metric=='ndcg':
            scores.append(ndcg_k_score(yss_true, yss_pred, k))
        elif eval_metric == 'map':
            scores.append(map_k_score(yss_true, yss_pred, k))
        elif eval_metric == 'mrr':
            scores.append(mrr_k_score(yss_true, yss_pred, k))
        else:
            scores.append(None)
    return scores

def evaluateLOOV(ys_true, yss_pred, eval_metrics, k=5):
    scores = []
    for eval_metric in eval_metrics:
        if eval_metric=='hr':
            scores.append(hr_k_score(ys_true, yss_pred, k))
        elif eval_metric=='arhr':
            scores.append(arhr_k_score(ys_true, yss_pred, k))
        else:
            scores.append(None)
    return scores



if __name__=='__main__':
    yss_true, yss_pred = [set([4,2]), set([3,1]), set([1])], [[3,1,2], [1,2], [2,3,1]]
    print(precision_k_score(yss_true, yss_pred, 3))
    print(recall_k_score(yss_true, yss_pred, 3))
    print(ndcg_k_score(yss_true, yss_pred, 3))

    # 1,1,0,1,1,1,0,0,1,0, 1,0,1,0,0,0,1,0,1,0
    yss_true, yss_pred = [set([0,1,3,4,5,8, 10,12,16,18])], [range(20)]
