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
npt = 5  # number of places in target

place_weights = {}
for i in range(npt):
    place_weights[i] = (1 / (i + 1))

print(place_weights)


lg = len(sub_files)
sub = [None] * lg
for i, file in enumerate(sub_files):
    ## input files ##
    print("Reading {}: w={} - {}".format(i, sub_weight[i], file))
    reader = csv.DictReader(open(file, "r"))    # 将csv文件数据读入到字典中
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
