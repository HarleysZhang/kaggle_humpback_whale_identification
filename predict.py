# encoding：utf-8
# filename: test.py
# function: 模型测试程序,

from my_utils.data_preprocess import *
from my_utils.input_data import *
from my_utils.model import *
# Read the dataset description
import gzip
# Read or generate p2h, a dictionary of image name to image id (picture to hash)
import pickle
import platform
import random
# Suppress annoying stderr output when importing keras.
import sys, os
from lapjv import lapjv
from math import sqrt
# Determine the size of each image
from os.path import isfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as pil_image
from imagehash import phash
from pandas import read_csv
from scipy.ndimage import affine_transform
from tqdm import tqdm
import time

from keras import backend as K
from keras import regularizers
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, \
	Lambda, MaxPooling2D, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import Sequence
from keras.models import Sequential, load_model

def score_reshape(score, x, y=None):
	"""
    将packed matrix的'得分'转换为方阵。
    @param score: the packed matrix
    @param x: 第一张图像的特征张量
    @param y: 第二张图像的张量，如果与x不同
    @结果为方阵
	"""
	if y is None:
		# When y is None, score is a packed upper triangular matrix.
		# Unpack, and transpose to form the symmetrical lower triangular matrix.
		m = np.zeros((x.shape[0], x.shape[0]), dtype=K.floatx())
		m[np.triu_indices(x.shape[0], 1)] = score.squeeze()
		m += m.transpose()
	else:
		m = np.zeros((y.shape[0], x.shape[0]), dtype=K.floatx())
		iy, ix = np.indices((y.shape[0], x.shape[0]))
		ix = ix.reshape((ix.size,))
		iy = iy.reshape((iy.size,))
		m[iy, ix] = score.squeeze()
	return m

def prepare_submission(threshold, filename):
	"""
	Generate a Kaggle submission file.
	@param threshold the score given to 'new_whale'
	@param filename the submission file name
	"""
	vtop = 0
	vhigh = 0
	pos = [0, 0, 0, 0, 0, 0]
	with open(filename, 'wt', newline='\n') as f:
		f.write('Image,Id\n')
		for i, p in enumerate(tqdm(submit)):
			t = []
			s = set()
			a = score[i, :]
			for j in list(reversed(np.argsort(a))):
				h = known[j]
				if a[j] < threshold and new_whale not in s:
					pos[len(t)] += 1
					s.add(new_whale)
					t.append(new_whale)
					if len(t) == 5: break;
				for w in h2ws[h]:
					assert w != new_whale
					if w not in s:
						if a[j] > 1.0:
							vtop += 1
						elif a[j] >= threshold:
							vhigh += 1
						s.add(w)
						t.append(w)
						if len(t) == 5: break;
				if len(t) == 5: break;
			if new_whale not in s: pos[5] += 1
			assert len(t) == 5 and len(s) == 5
			f.write(p + ',' + ' '.join(t[:5]) + '\n')
	return vtop, vhigh, pos


# Find elements from training sets not 'new_whale'
tic = time.time()
h2ws = {}
for p, w in tagged.items():
	if w != new_whale:  # Use only identified whales
		h = p2h[p]
		if h not in h2ws: h2ws[h] = []
		if w not in h2ws[h]: h2ws[h].append(w)
known = sorted(list(h2ws.keys()))

# Dictionary of picture indices
h2i = {}
for i, h in enumerate(known): h2i[h] = i

model, branch_model, head_model = build_model(64e-5, 0)
model.summary()

# Load the model
if isfile('./models/standard_epoch350.h5'):
	model.load_weights('./models/standard_epoch350.h5')
	print('Load model success!')
else:
	print('The model file is not exist!')

# Evaluate the model.
fknown = branch_model.predict_generator(FeatureGen(known), max_queue_size=20, workers=10, verbose=0)
fsubmit = branch_model.predict_generator(FeatureGen(submit), max_queue_size=20, workers=10, verbose=0)
score = head_model.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=10, verbose=0)
score = score_reshape(score, fknown, fsubmit)

# Generate the subsmission file.
prepare_submission(0.99, './submissions/submission.csv')
toc = time.time()
print("Submission time: %.3f minute" % (toc - tic) / 60.)

