import HCModel
import pandas as pd
import numpy as np
from fastparquet import ParquetFile
import sklearn
import random
import pickle
import time
Cats_description = pd.read_csv('categories_tree.csv')

Cat_tree = HCModel.Cat_tree(Cats_description)


pf = ParquetFile('train.parquet')
columns = ['title', 'short_description', 'name_value_characteristics', 'rating',
           'feedback_quantity', 'category_id']
Data_pd = pf.to_pandas(columns=columns)


def pre_transofrm(data):
    data_text = np.empty((data.shape[0],), dtype=object)
    for i in range(data.shape[0]):
        data_text[i] = data['title'][i] + ' '
        if not data['short_description'][i] is None:
            data_text[i] += data['short_description'][i] + ' '
        if not data['name_value_characteristics'][i] is None:
            data_text[i] += data['name_value_characteristics'][i]
    data['entire_text'] = data_text


def get_hF(predicted_classes, target_classes, classes_tree):
    len_cross = 0
    len_p = 0
    len_t = 0
    for i in range(len(predicted_classes)):
        pred_class_expanded = []
        target_class_expanded = []

        tmp = predicted_classes[i]
        while tmp != 0:
            pred_class_expanded.append(tmp)
            tmp = classes_tree[tmp]

        tmp = target_classes[i]
        while tmp != 0:
            target_class_expanded.append(tmp)
            tmp = classes_tree[tmp]

        len_of_eq = 1
        flag = 1
        while len_of_eq <= min(len(pred_class_expanded),
                               len(target_class_expanded)) and flag:
            if pred_class_expanded[-len_of_eq] == target_class_expanded[-len_of_eq]:
                len_of_eq += 1
            else:
                flag = 0

        len_cross += len_of_eq - 1
        len_p += len(pred_class_expanded)
        len_t += len(target_class_expanded)

    return 2 * (len_cross) / (len_t + len_p)


pre_transofrm(Data_pd)

classes_dict = dict()

for item in Cats_description.itertuples(index=False):
    classes_dict[int(item[0])] = int(item[2])

Alpha = [1/4, 1/2, 3/4]

Beta = [100,  500]

N_est = [50, 100]

Max_feat = [3000, 15000]

Using_data = Data_pd.iloc[:100000]
np.random.seed(31415)

Train_data, Test_data = sklearn.model_selection.train_test_split(Data_pd, test_size=0.2)
'''
CV_data = []

Train = Train_data

for i in range(4):
    Train, Test = sklearn.model_selection.train_test_split(Train, test_size=1/(5-i))
    CV_data.append(Test)

CV_data.append(Train)

random.seed = 42

test_i = random.randrange(5)
test = CV_data[test_i]
train = pd.concat([CV_data[i] for i in range(5) if i != test_i])
'''
file = open('model', 'wb')
data_tree = HCModel.Data_tree(Train_data, Cat_tree.root)
tic = time.perf_counter()
classifier = HCModel.HierarchicalClassifier(data_tree, 2,
                                            Beta[0], 0.8,
                                            N_est[1])
toc = time.perf_counter()

pickle.dump(classifier,file)

print(f"Вычисление заняло {toc - tic:0.4f} секунд")
'''
tmp = [classifier.root]
    
for i in range(5):
    array = []
    s = ''
    for node in tmp:
        if type(node) != HCModel.End and len(node.u_w):
            s += str(len(node.c_d)) + '/' + str(len(node.tokenizer.word_index)) + ' '
            array.extend(node.next_nodes)
    print(s)
    tmp = array
'''
target = Test_data.category_id.to_numpy()
predicted = [int(classifier.predict(text)) for text in Test_data.entire_text]

print(get_hF(predicted, target, classes_dict))
