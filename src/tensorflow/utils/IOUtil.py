#!/usr/bin/python

import csv
import pandas
from scipy.sparse import lil_matrix

from src.utils.Util import split_row

def loadA2BFromCSV(fpath):
	A2B = []
	with open(fpath, 'r', newline='') as csvf:
		csvreader = csv.reader(csvf)  # 创建reader对象
		for line in csvreader:
			a, b = line[0].strip(), line[1].strip()
			A2B.append(a+'_'+b)
	A2B = list(set(A2B))
	A2B_ = []
	for a_b in A2B:
		phs = a_b.split('_')
		a, b = int(phs[0].strip()), int(phs[1].strip())
		A2B_.append([a, b])
	return A2B_

# load the rating data as dok_matrix
def loadSparseR(usernum, itemnum, inFilePath):
	sR = lil_matrix((usernum, itemnum))

	# data = pandas.read_csv(inFilePath, header=None, sep='\t').values
	# for line in data:
	# 	sR[int(line[0]), int(line[1])] = 1

	with open(inFilePath, 'r') as infile:
		for line in infile.readlines():
			phs = split_row(line)
			sR[int(phs[0]), int(phs[1])] = 1
	return sR

def saveTuples(tuples, outFilePath):
	with open(outFilePath, 'w') as outfile:
		for ind in range(len(tuples)):
			a, b = tuples[ind]
			outfile.write(str(a)+'\t'+ str(b)+'\n')

if __name__ == '__main__':
	data_dir = '../../data/'

	# A2B = loadA2BFromCSV(data_dir+'kanzhun/exp2job.csv')
	# print(len(A2B))

	# kanzhun
	loadSparseR(799, 993, data_dir+'kanzhun/E2J.txt')
	loadSparseR(993, 799, data_dir+'kanzhun/J2E.txt')

	# libimseti
	# loadSparseR(9066, 9072, data_dir+'libimseti/F2M.txt')
	# loadSparseR(9072, 9066, data_dir+'libimseti/M2F.txt')
