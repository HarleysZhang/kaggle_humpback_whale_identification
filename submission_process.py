# coding: utf-8
# filename: submission_process.py
# function: 处理具有两个id的鲸鱼图像测试文件


import pandas as pd
import os

duplicate_whale = {
                    'w_84b1b28': 'w_163b2b8',
                    'w_297e53a': 'w_9f2cfff',
                    'w_b43d0fa': 'w_19f507e',
                    'w_e4a9205': 'w_5c4c166',
                    'w_5fcf350': 'w_e21b25f',
                    'w_bd18f3c': 'w_185044d',
                    'w_3d7bc85': 'w_cb0e8c2',
                    'w_fae4be5': 'w_f84e6e1',
                    'w_2ac2bac': 'w_3bf887a',
                    'w_2f8de4f': 'w_f7234f1',
                    'w_7efe670': 'w_597758e',
                    'w_28218e1': 'w_62bb54b',
                    'w_c1b90d1': 'w_950c471',
                    'w_94e8c02': 'w_169253a',
                    'w_eb31833': 'w_662c89b',
                    'w_a44a9b2': 'w_6fe3566',
                    'w_5ce4848': 'w_81ed5f8',
                    'w_b2ef717': 'w_07d4b0a',
                    'w_808d40b': 'w_b035775',
                    'w_c97ad94': 'w_a3181a0',
                    'w_b82d0eb': 'w_aa0dacc',
                    'w_298f605': 'w_43614bc',
                    'w_bbf0c38': 'w_32e705d',
                    'w_7d46ac3': 'w_3729146',
                    'w_25d662d': 'w_d2facb4',
                    'w_5010896': 'w_a1a0090',
                    'w_8431ae8': 'w_115e880',
                    'w_f5d4627': 'w_2623921',
                    'w_b6e4761': 'w_bfecb74',
                    'w_52162eb': 'w_a112639',
                    'w_60b027a': 'w_c87817e',
                    'w_022b708': 'w_d215a68',
                    'w_da651f6': 'w_d92b4c4',
                    'w_5072c08': 'w_8694cb5',
                    'w_f94fc92': 'w_228c7ee',
                    'w_0532483': 'w_444b771',
                    'w_0cc0430': 'w_97074d8',
                    'w_d7aad03': 'w_e56d7b2',
                    'w_6525e6b': 'w_f1b016e',
                    'w_ddda4a4': 'w_9353e7c',
                    'w_ea2d2f9': 'w_3789389',
                    'w_26ba5fd': 'w_8daccba',
                    'w_792ffd7': 'w_a39b2e2',
                    'w_e12b78d': 'w_9ffc9aa',
                    'w_353d249': 'w_af1d57b',
                    'w_4130233': 'w_cc93694',
                    'w_c1715f5': 'w_706e4fb',
                    'w_ac6aee2': 'w_a8276b4',
                    'w_c7e1b12': 'w_62b631e',
                    'w_c543f7c': 'w_2d1d67a',
                    'w_a335fc2': 'w_c99807e',
                    'w_f119baa': 'w_3f213d5',
                    'w_2f350be': 'w_6f22173',
                    'w_954fec8': 'w_0ee7878',
                    'w_27272a5': 'w_7e5b9da',
                    'w_b5e6c9c': 'w_0a0c768',
                    'w_e20dcb5': 'w_6822dbc',
                    'w_92a47eb': 'w_ea6a9f4',
                    'w_5a3e0de': 'w_0b398b2',
                    'w_53ce2e1': 'w_59109e5',
                    'w_f91389a': 'w_7d8c37c',
                    'w_1d532b4': 'w_b54f70f',
                    'w_169a302': 'w_feb4051',
                    'w_596f6bc': 'w_d588fb4',
                    'w_2ed0d2f': 'w_7537a43',
                    'w_4e00a18': 'w_6c2aa5f',
                    'w_012678c': 'w_f497e3f',
                    'w_6eb10d2': 'w_2e5ad54',
                    'w_20950a9': 'w_1b2bf0f',
                    'w_993320b': 'w_6dbfd7a',
                    'w_21fd105': 'w_04de239',
                    'w_4135cb8': 'w_8d4c9f7',
                    'w_1d35a02': 'w_f84e6e1',
                    'w_cded134': 'w_a6df3a5'
                    }

src_dir = os.getcwd()
src_dir = src_dir.replace('\\', '/')

SUB_FILE = os.path.join(src_dir, "submissions", "submission_leak1_2.csv")
SUB_DF = pd.read_csv(SUB_FILE)

print(SUB_DF.head())

label_list = [ll for _, _, ll in SUB_DF.to_records()]
# print(label_list[:2])
# print(type(label_list))
# print(len(label_list))
# print(type(SUB_DF['Id']))

# 将submission文件'Id'列元素从字符串转为列表, 方便后续获取索引
labels = []
for label in label_list:
    label = label.split()
    labels.append(label)
# print(labels[:2])

num = 0
for i, label in enumerate(labels):
    new_la = []
    for ii, la in enumerate(label):
        new_la.append(la)
        for key, values in duplicate_whale.items():
            if key == la:
                num += 1
                new_la.append(values)
            if len(new_la) >= 5:
                break
            if values == la:
                num += 1
                new_la.append(key)
            if len(new_la) >= 5:
                break
    labels[i] = new_la

print('There are %d id need to be replaced' % num)
# print(labels)

# 将submission文件'Id'列元素从列表转为字符串
for i, str_ll in enumerate(labels):
    new_la = ''
    for ii, la in enumerate(str_ll):
        if ii == 0:
            new_la = new_la + la
        else:
            new_la = new_la + ' ' + la
    # print(type(new_la))
    labels[i] = new_la

SUB_DF['Id'] = labels
print(SUB_DF.head())
SUB_DF.to_csv('./submissions/duplicate_sub.csv', index=None)
print('duplicate process success!')

