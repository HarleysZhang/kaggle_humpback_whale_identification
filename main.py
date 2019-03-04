# coding:utf-8
# filename:main.py
# function:鲸鱼识别程序,模型训练和模型测试,主程序


from data_preprocess import *
# from input_data import *
from model import *
from keras.callbacks import TensorBoard
import keras


def set_lr(model, lr):
	"""
	set the lr of model
	"""
	K.set_value(model.optimizer.lr, float(lr))


def get_lr(model):
	"""
	return the lr of model
	"""
	return K.get_value(model.optimizer.lr)


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
		iy, ix = np.indices((y.shape[0], x.shape[0]))    # np.indices()将创建一组数组（堆积为一个更高维的数组）
		ix = ix.reshape((ix.size,))
		iy = iy.reshape((iy.size,))
		m[iy, ix] = score.squeeze()                      # 从数组的形状中删除单维条目, 即把shape中为1的维度去掉
	return m


def compute_score(verbose=1):
	"""
	Compute the score matrix by scoring every pictures from the training set against every other picture O(n^2).
	"""
	features = branch_model.predict_generator(FeatureGen(train, verbose=verbose), max_queue_size=12, workers=6,
											  verbose=0)
	score = head_model.predict_generator(ScoreGen(features, verbose=verbose), max_queue_size=12, workers=6, verbose=0)
	score = score_reshape(score, features)
	return features, score


def make_steps(step, ampl):
	"""
	执行训练
	@param step: 训练的epoch数。
	@param ampl: K值, score matrix的随机分量。
	"""
	global w2ts, t2i, steps, features, score, histories

	# 打乱训练图片
	random.shuffle(train)

	# 将鲸鱼id映射到相关的训练图片的hash表上去。
	w2ts = {}
	for w, hs in w2hs.items():
		for h in hs:
			if h in train_set:
				if w not in w2ts: w2ts[w] = []
				if h not in w2ts[w]: w2ts[w].append(h)
	for w, ts in w2ts.items(): w2ts[w] = np.array(ts)

	# 将训练图片hash值映射到'train'数组中的索引  
	t2i = {}
	for i, t in enumerate(train): t2i[t] = i

	# 计算每个图片对的匹配分数
	features, score = compute_score()

	# callback TensorBoard_class
	tbCallBack = TensorBoard(
		log_dir='./logs',
		histogram_freq=0,
		write_graph=True,
		write_images=True)
	# 训练模型'step'个epoch
	history = model.fit_generator(
		TrainingData(score + ampl * np.random.random_sample(size=score.shape), steps=step, batch_size=2),
		initial_epoch=steps,
		epochs=steps + step,
		max_queue_size=12,
		workers=6,
		verbose=2,
		callbacks=[tbCallBack]).history
	steps += step
	print('The steps(epochs) is', steps)
	# 收集历史数据
	history['epochs'] = steps
	history['ms'] = np.mean(score)
	history['lr'] = get_lr(model)
	print('The epochs lr and ms of history is', history['epochs'], history['lr'], history['ms'])
	histories.append(history)

	# 将历史数据写入文本保存
	log = open('model_log.txt', "a")
	log_str = '.'
	his_loss, his_bc, his_acc = history['loss'], history['binary_crossentropy'], history['acc']
	for ep in range(len(his_loss)):
		log_str += 'Epoch %d/%d - ' % (steps - len(his_loss) + 1 + ep, steps)
		log_str += 'loss: %.4f - binary_crossentropy: %.4f - acc: %.4f\n' % (his_loss[ep], his_bc[ep], his_acc[ep])
	log_str += 'Epoch %d, lr: %.5f' % (history['epochs'], history['lr'])
	log_str += '\n'
	log.write(log_str)  # 将字符串以'a'模式写入文本文件


# ---------------------------------------Start perform training model----------------------------------------
model_name = 'standard'
histories = []
steps = 0

if isfile('./mpiotte-standard.model'):
	tmp = keras.models.load_model('./mpiotte-standard.model')
	model.set_weights(tmp.get_weights())
else:
	log = open('model_log.txt', "a")
	start_str = '[\nSiamese===========lapjv==========Size512*3==========Epoch450============MultiThreads]\n'
	log.write(start_str)
	log.write('\nEpoch====================================================================>10\n')
	print('Start information write success!')
	# epoch -> 10
	make_steps(10, 1000)
	ampl = 100.0
	for _ in range(2):
		print('noise ampl.  = ', ampl)
		make_steps(5, ampl)
		ampl = max(1.0, 100 ** -0.1 * ampl)
	log.write('\nEpoch====================================================================>110\n')
	# epoch -> 110
	for _ in range(18): make_steps(5, 1.0)
	model.save('standard_epoch110.model')
	log.write('\nEpoch===================================================================>160\n')
	# epoch -> 160
	set_lr(model, 16e-5)
	for _ in range(10): make_steps(5, 0.5)
	model.save('standard_epoch160.model')
	log.write('\nEpoch===================================================================>200\n')
	# epoch -> 200
	set_lr(model, 4e-5)
	for _ in range(8): make_steps(5, 0.25)
	model.save('standard_epoch200.model')
	log.write('\nEpoch===================================================================>210\n')
	# epoch -> 210
	set_lr(model, 1e-5)
	for _ in range(2): make_steps(5, 0.25)
	model.save('standard_epoch210.model')
	# epoch -> 220
	for _ in range(2): make_steps(5, 0.25)
	model.save('standard_epoch220.model')
	# epoch -> 230
	for _ in range(2): make_steps(5, 0.25)
	model.save('standard_epoch230.model')
	# epoch -> 240
	for _ in range(2): make_steps(5, 0.25)
	model.save('standard_epoch240.model')
	# epoch -> 250
	for _ in range(2): make_steps(5, 0.25)
	model.save('standard_epoch250.model')
	log.write('\nEpoch===================================================================>300\n')
	# epoch -> 300
	weights = model.get_weights()
	model, branch_model, head_model = build_model(64e-5, 0.0002)
	model.set_weights(weights)
	for _ in range(10): make_steps(5, 1.0)
	model.save('standard_epoch300.model')
	log.write('\nEpoch===================================================================>350\n')
	# epoch -> 350
	set_lr(model, 16e-5)
	for _ in range(10): make_steps(5, 0.5)
	model.save('standard_epoch350.model')
	log.write('\nEpoch===================================================================>390\n')
	# epoch -> 390
	set_lr(model, 4e-5)
	for _ in range(8): make_steps(5, 0.25)
	model.save('standard_epoch390.model')
	log.write('\nEpoch===================================================================>400\n')
	# epoch -> 400
	set_lr(model, 1e-5)
	for _ in range(2): make_steps(5, 0.25)
	model.save('standard_epoch400.model')
	# epoch -> 410
	set_lr(model, 1e-5)
	for _ in range(2): make_steps(5, 0.25)
	model.save('standard_epoch410.model')
	# epoch -> 420
	for _ in range(2): make_steps(5, 0.25)
	model.save('standard_epoch420.model')
	# epoch -> 430
	for _ in range(2): make_steps(5, 0.25)
	model.save('standard_epoch430.model')
	# epoch -> 440
	for _ in range(2): make_steps(5, 0.25)
	model.save('standard_epoch440.model')
	log.write('\nEpoch===================================================================>450\n')
	# epoch -> 450
	model.save('standard.model')


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

# Evaluate the model.
fknown = branch_model.predict_generator(FeatureGen(known), max_queue_size=20, workers=10, verbose=0)
fsubmit = branch_model.predict_generator(FeatureGen(submit), max_queue_size=20, workers=10, verbose=0)
score = head_model.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=10, verbose=0)
score = score_reshape(score, fknown, fsubmit)

# Generate the subsmission file.
prepare_submission(0.99, './submission.csv')
toc = time.time()
print("Submission time: ", (toc - tic) / 60.)
