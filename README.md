# [Humback Whale Identification](https://www.kaggle.com/c/humpback-whale-identification/discussion)
## Some best soluton
|rank    |solution                |github            |author               |keyword        |
|---------|------------------------|-------------------|--------------------|------------------|
|1th|[1th Place Solution](https://www.kaggle.com/c/humpback-whale-identification/discussion/82366)|[Github code](https://github.com/earhian/Humpback-Whale-Identification-1st-)|earhian|classification|
|3rd|[3rd Place Solution](https://www.kaggle.com/c/humpback-whale-identification/discussion/82484)|Github|pudae|ArcFace|
|4th|[4th Place Solution](https://www.kaggle.com/c/humpback-whale-identification/discussion/82356)|Github code|David|SIFT+Siamese|
|7th|[7th Place Solution](https://www.kaggle.com/c/humpback-whale-identification/discussion/82352)|[Github code](https://github.com/ducha-aiki/whale-identification-2018)|old-ufo|classification|
|9th|[9th Place Solution](https://www.kaggle.com/c/humpback-whale-identification/discussion/82427)|Github code|lvan Sosin|[GapNet](https://openreview.net/forum?id=ryl5khRcKm)|
|25th|[25th Place Solution](https://www.kaggle.com/c/humpback-whale-identification/discussion/82409)|Github code|Bartek|CosFace+ProtoNets|
|31st|[31st Place Solution](https://www.kaggle.com/c/humpback-whale-identification/discussion/82393)|[Github code](https://github.com/suicao/Siamese-Whale-Identification)|Khoi Nguyen|RGB|
|57th|[57th Place Solution](https://www.kaggle.com/c/humpback-whale-identification/discussion/82364)|Github code|Miguel Pinto|SoftTripletLoss|

## My solution

Heavily based on [Whale Recognition Model with score 0.78563](https://www.kaggle.com/martinpiotte/whale-recognition-model-with-score-0-78563)

### Training

+ Framework: `Keras(backend: tensorflow)`
+ Model: `Siamese(CNN+Metric Learning)`
+ Augmentation: `slight(otation, shear, height_zoom, width_zoom, height_shift, width_shift)`
+ Preprocess: `rotate some special images, convert grayscale,get bounding boxs, affine tranformation`
+ Optimizer: Adam
+ Learning rate:  `start at 64e-5, and 4 times less training per epoch group`
+ Image size: `512*512`
+ Epochs: `400 or more`
+ Batch size: `32`

### Prediction

+ Threshold: `0.99 and 0.94 with bootstrapping`
+ TTA number: `4`
+ TTA augmentaion: `random slight: (rotation, shear, height_zoom, width_zoom, height_shift, width_shift)`

### Result

+ Training takes about more than 80 hours on GTX 1080TI without pretrained state-of-art model
+ Public LB: `0.92248`
+ Private LB: `0.92761`

### Mode result ensemble:

+ Ensemble of ensemble is not feasible, but ensemble is very effective
+ If single model is selected as far as possible for fusion, the effect is better, but the model difference is large, so the fusion effect is better. The fusion effect of models with similar Epochs is not as good as that with large difference
+ The ensemble of tta*4 + original result is effective

#### ensemble code

```python
# coding:utf-8
# filename:ensemble.py
# function:模型识别结果融合程序,融合4个最好的结果

import csv
sub_files = [
            './submissions/submission_Simaese_Epochs220_multithreads_lapjv_512size_0.883.csv',
            './submissions/submission_Simaese_Epochs210_multithreads_lapjv_384size_0.884.csv',
            './submissions/submission_ensemble_(Epoch250_tta*4+original)_0.901.csv',
            './submissions/submission_Simaese_Epochs390_multithreads_lapjv_512size_0.905.csv',
            './submissions/submission_ensemble_(Boot_Epoch350_tta*4+original)_0.908.csv',
            './submissions/submission_ensemble_(Epoch400_tta*4+original)_0.912.csv']

print(len(sub_files))

# Weights of the individual subs
sub_weight = [
            0.883 ** 2,
            0.884 ** 2,
            0.901 ** 2,
            0.905 ** 2,
            0.908 ** 2,
            0.912 ** 2]
Hlabel = 'Image'
Htarget = 'Id'
npt = 5 # number of places in target
place_weights = {}
for i in range(npt):
    place_weights[i] = (1 / (i + 1))
print(place_weights)
lg = len(sub_files)
sub = [None] * lg
for i, file in enumerate(sub_files):
    ## input files ##
    print("Reading {}: w={} - {}".format(i, sub_weight[i], file))
    reader = csv.DictReader(open(file, "r")) # 将csv文件数据读入到字典中
    sub[i] = sorted(reader, key=lambda d: str(d[Hlabel]))
## output file ##
out = open("./submissions/submission_ensemble_zh.csv", "w", newline='')
writer = csv.writer(out)
writer.writerow([Hlabel, Htarget])
for p, row in enumerate(sub[0]):
    target_weight = {}
    for s in range(lg):
        row1 = sub[s][p]
        for ind, trgt in enumerate(row1[Htarget].split(' ')):
            target_weight[trgt] = target_weight.get(trgt, 0) + (place_weights[ind] * sub_weight[s])
    tops_trgt = sorted(target_weight, key=target_weight.get, reverse=True)[:npt]
    writer.writerow([row1[Hlabel], " ".join(tops_trgt)])
out.close()
```

## My conclusion

### Work

+ Large image size helps a lot
+ ensemble is useful, but correct ensemble strategy is more useful
+ TTA maybe help, but ensemble of tta must be help
+ Put all images into SSD faster than HDD in training
+ training more epochs helps a lot
+ bootstrapping helps, but it need more time to train

### Don't work

+ pure classition don't work, but if you do some extra works,classition maybe very useful, such as this [1thsolution](https://www.kaggle.com/c/humpback-whale-identification/discussion/82366)
+ n-fold CV: my parteners have tried 5-fold CV, but it dont't work, maybe our ways have some problem, but i dont see n-fold CV as solution in [Kaggle Dissussion](https://www.kaggle.com/c/humpback-whale-identification/discussion)

### Uncertain

+ Grayscale images are not necessarily more effective than RGB

## Usage

### Environments

##### Hardware requirements

+ GTX1060, GTX1080TI better
+ 32GB Memory
+ SSD 

#### Software requirments

+ Ubuntu 18.04
+ Anaconda3/Python3
+ Keras(backend: tensorflow

### Steps for usage

+ 1.clone the repository

```shell
git https://github.com/HarleysZhang/kaggle_humpback_whale_identification.git
cd kaggle_humpback_whale_identification
```

+ 2.install requirements

```shell
pip3 install -r requirements.txt
```

+ 3.download data  and copy it to data folder

```shell
kaggle competitions download -c humpback-whale-identification
```

```shell
cp train ./data/
cp test ./data/
cp train.csv ./data/
cp sample_submission.csv ./data/
```

+ 4.train your model
without bootstrapping

```shell
python3 main_all.py
```

with bootstrapping

```shell
python3 main_with_bootstrapping.py
```

+ 5.ensemble submission file

```shell
python test.py
# python test_tta.py    # with tta
```

## Some Code Interpretation

Build a transformation matrix with the specified characteristics.

```python
def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
	"""
	Build a transformation matrix with the specified characteristics.
	"""
	rotation = np.deg2rad(rotation)
	shear = np.deg2rad(shear)
	rotation_matrix = np.array(
		[[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
	shift_matrix = np.array(
		[[1, 0, height_shift], [0, 1, width_shift], [0, 0, 1]])
	shear_matrix = np.array(
		[[1, np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
	zoom_matrix = np.array(
		[[1.0 / height_zoom, 0, 0], [0, 1.0 / width_zoom, 0], [0, 0, 1]])
	shift_matrix = np.array(
		[[1, 0, -height_shift], [0, 1, -width_shift], [0, 0, 1]])
	return np.dot(np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix, shift_matrix))
```

Compute the score matrix by scoring every pictures from the training set against every other picture O(n^2) with multithreads.

```python
def compute_score(verbose=1):
	"""
	Compute the score matrix by scoring every pictures from the training set against every other picture O(n^2).
	"""
	features = branch_model.predict_generator(FeatureGen(train, batch_size=64, verbose=verbose),max_queue_size=12, workers=6, verbose=0)
	num_threads = 6
	batch = features.shape[0] // (num_threads - 1)
	if features.shape[0] % batch <= 3:
	            num_threads = 5
		if features.shape[0] % batch is not 0:
			batch += 1
	all_score = []
	for start in range(0, features.shape[0], batch):
		end = min(features.shape[0], start + batch)
		temp_features = features[start:end, :]
		temp_score = head_model.predict_generator(ScoreGen(temp_features, batch_size=4096, verbose=verbose)，max_queue_size=12, workers=6, verbose=0)
		temp_score = score_reshape(temp_score, temp_features)
		all_score.append(temp_score)
	score = np.zeros((features.shape[0], features.shape[0]), dtype=K.floatx())
	for i, start in enumerate(range(0, features.shape[0], batch)):
		end = min(features.shape[0], start + batch)
		score[start:end, start:end] = all_score[i]
	return features, score
```

sompute Linear programming problem with multithreads

```python
def my_lapjv(score):
    num_threads = 6
    batch = score.shape[0] // (num_threads - 1)
    if score.shape[0] % batch <= 3:
        num_threads = 5
        if score.shape[0] % batch is not 0:
            batch += 1
    # print(batch)
    tmp = num_threads * [None]
    threads = []
    thread_input = num_threads * [None]
    thread_idx = 0
    for start in range(0, score.shape[0], batch):
        end = min(score.shape[0], start + batch)
        # print('%d %d' % (start, end))
        thread_input[thread_idx] = score[start:end, start:end]
        thread_idx += 1

    def worker(data_idx):
        x, _, _ = lapjv(thread_input[data_idx])
        tmp[data_idx] = x + data_idx * batch

    # print("Start worker threads")
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        if t is not None:
            t.join()
    x = np.concatenate(tmp)
    # print("LAP completed")
    return x
```

## Reference

[Whale Recognition Model with score 0.78563](https://www.kaggle.com/martinpiotte/whale-recognition-model-with-score-0-78563)
