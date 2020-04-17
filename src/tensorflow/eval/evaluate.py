import datetime as dt
from src.metrics.ranking import evaluateCV, evaluateLOOV

class Evaluator(object):
    def __init__(self, split_method, eval_metrics, topNs):
        self.split_method = split_method
        self.eval_metrics = eval_metrics
        self.topNs = topNs

    def eval(self, yss_true, yss_pred):
        print('\t', dt.datetime.now().strftime('%m-%d %H:%M:%S'), 'start eval()')

        scoress = []
        # print(yss_true[0], yss_pred[0])
        if self.split_method == 'cv':
            for topN in self.topNs:
                scoress.append(evaluateCV(yss_true, yss_pred, self.eval_metrics, topN))
        elif self.split_method == 'loov':
            for topN in self.topNs:
                scoress.append(evaluateLOOV(yss_true, yss_pred, self.eval_metrics, topN))
        return scoress
