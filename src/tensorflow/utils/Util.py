
import numpy as np

# split the row content
def split_row(row_content):
	if ',' in row_content:
		return row_content.strip().split(',')
	if ';' in row_content:
		return row_content.strip().split(';')
	else:
		return row_content.strip().split()


# binary (sparse matrix)
def matBinarize(sR, r_threshold):
	return (sR>r_threshold).astype(np.float32)



def avestd(eval_metrics, scoresss, topNs):
	print(eval_metrics)
	for i in range(len(topNs)):
		aves, stds = [], []
		for j in range(len(eval_metrics)):
			ave = scoresss[:, i, j].sum(0) / len(scoresss)
			std = np.sqrt(np.power(np.array(scoresss[:, i, j]) - ave, 2).sum(0) / len(scoresss))
			aves.append(ave)
			stds.append(std)
		print(str(topNs[i])+':', 'ave=[' + ','.join(['%.4f' % ave for ave in aves]) + ']',
              'std=[' + ','.join(['%.4f' % std for std in stds]) + ']' if len(scoresss)>1 else '')