import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
import scipy.sparse as sparse
from sklearn.ensemble import RandomForestClassifier
import random

pd.options.mode.chained_assignment = None


class Node:
    def __init__(self, data, cat=0, n=0):
        self.data = data
        self.next_nodes = np.empty(n, dtype=Node)
        self.cat = cat

class Cat_node:
    def __init__(self, next_n=[], cat=0):
        self.next_nodes = next_n
        self.cat = cat


class Cat_tree:

    def __init__(self, tree):
        nodes_dict = dict()

        for item in tree.itertuples(index=False):
            if item[0] not in nodes_dict:
                nodes_dict[item[0]] = Cat_node([], item[0])
            if item[2] in nodes_dict:
                nodes_dict[item[2]].next_nodes.append(nodes_dict[item[0]])
            else:
                nodes_dict[item[2]] = Cat_node([nodes_dict[item[0]]], item[2])

        self.root = nodes_dict[0]



class Data_tree:
    @staticmethod
    def tree_desent(node, cat_list):
        if not len(node.next_nodes):
            cat_list.append(node.cat)
        else:
            for item in node.next_nodes:
                Data_tree.tree_desent(item, cat_list)

    @staticmethod
    def get_cats_from_node(cat_node):
        cats = []
        Data_tree.tree_desent(cat_node, cats)
        return cats

    @staticmethod
    def data_for_node(tree_node, parent):
        cats = Data_tree.get_cats_from_node(tree_node)
        return parent.data.loc[parent.data.category_id.isin(cats)]

    @staticmethod
    def build_data_tree(cur_data_node, cur_cat_node):
        for i, item in enumerate(cur_cat_node.next_nodes):
            tmp = Node(Data_tree.data_for_node(item, cur_data_node), item.cat, len(item.next_nodes))
            cur_data_node.next_nodes[i] = tmp
            Data_tree.build_data_tree(tmp, item)

    def __init__(self, data, cat_tree_root):
        self.root = Node(data, 0, len(cat_tree_root.next_nodes))
        Data_tree.build_data_tree(self.root, cat_tree_root)


class RandomClassifier:
    def __init__(self, num):
        self.num_classes = num

    def fit(self, a):
        pass

    def predict(self, a):
        return [random.randrange(self.num_classes)]


class End:
    next_nodes = []
    u_w = []

    def __init__(self, data_node):

        self.cat = data_node.cat
        self.data_len = len(data_node.data)

    def fit(self):
        pass

    def predict(self):
        return self.cat


class NodeClassifier:

    def get_u_w(self, alpha, max_feat, data_node):
        num_words = len(self.tokenizer.word_index)
        cats_num = len(data_node.next_nodes)
        n_w_num = len(self.n_words_indices)
        w_c = np.zeros(n_w_num)
        for text in data_node.data['entire_text']:
            w_c += self.tokenizer.texts_to_matrix([text],
                                                  mode='binary')[0][self.n_words_indices].astype(int)

        n_z_w = self.n_words_indices[w_c != 0]
        w_c = w_c[w_c != 0]
        w_d = np.zeros((len(w_c), cats_num))

        for cat in np.unique(self.target).astype(int):
            for text in data_node.data['entire_text'].to_numpy()[self.target == cat]:
                w_d[:, cat] += self.tokenizer.texts_to_matrix([text],
                                                              mode='binary')[0][n_z_w].astype(float)

            w_d[:, cat] /= w_c
        w_d.sort(axis=1)

        dispersions = np.sqrt(np.sum(np.flip(w_d, axis=1) * (np.arange(cats_num)**2)[np.newaxis],
                                         axis=1))
        first_step_i = dispersions < alpha*1

        first_step_i = np.arange(len(first_step_i), dtype=int)[first_step_i]
        first_step_w = n_z_w[first_step_i]

        self.c_d = first_step_w

        if len(first_step_w) > max_feat * num_words:
            second_step = np.argsort(w_c[first_step_i])[-int(max_feat*num_words):]
            return n_z_w[first_step_i[second_step]]

        return first_step_w

    def get_tf_idf(self, texts):
        return self.tokenizer.texts_to_matrix(texts, mode='tfidf')[:, self.u_w]

    def fut_extraction(self, alpha, max_feat, data_node):
        u_w = self.get_u_w(alpha, max_feat, data_node)
        self.u_w = u_w
        mode = 'numpy'

        train_m = None

        if len(data_node.data) > 2000:
            mode = 'sparse'

        if mode == 'numpy':
            train_m = self.get_tf_idf(data_node.data['entire_text']).astype(np.float32)

        elif mode == 'sparse':
            train_m = sparse.lil_matrix((len(data_node.data), len(u_w)), dtype=np.float32)

            for i in range(len(data_node.data)):
                train_m[i] = sparse.lil_matrix(self.get_tf_idf([
                    data_node.data['entire_text'].iloc[i]])[0].astype(np.float32),
                                                                dtype=np.float32)

        return train_m

    def get_n_w_i(self, tokenizer, n_words):
        return np.array([tokenizer.word_index[word] for word in tokenizer.word_counts.keys()
                         if word in n_words], dtype=int)

    def make_target(self, data_node):
        cats = []
        for node in data_node.next_nodes:
            cats.append(np.unique(node.data['category_id']))
        target = np.zeros(len(data_node.data))
        for i, cat in enumerate(cats):
            target[data_node.data.category_id.isin(cat)] = i
        return target

    def __init__(self):
        self.data_len = 0
        self.u_w = []
        self.c_d = 0
        self.next_nodes = []
        self.model = None
        self.tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n«»0123456789')
        self.n_words_indices = []
        self.target = []

    def fit(self, alpha, beta, max_feat, n_est, data_node, n_words):
        self.data_len = len(data_node.data)
        self.tokenizer.fit_on_texts(data_node.data['entire_text'].to_numpy())
        if len(data_node.data) >= beta:
            self.target = self.make_target(data_node)
            self.n_words_indices = self.get_n_w_i(self.tokenizer, n_words)
            train = self.fut_extraction(alpha, max_feat, data_node)
            if train.shape[1] != 0:
                self.model = RandomForestClassifier(n_estimators=n_est)
                self.model.fit(train, self.make_target(data_node))
            else:
                self.model = RandomClassifier(len(data_node.next_nodes))
            del train
        else:
            self.model = RandomClassifier(len(data_node.next_nodes))

    def predict(self, text):
        result = int(self.model.predict(self.get_tf_idf([text]))[0])
        return result


class HierarchicalClassifier:

    alphabet = ["а", "б", "в", "г", "д", "е", "ё", "ж",
                "з", "и", "й", "к", "л", "м", "н", "о",
                "п", "р", "с", "т", "у", "ф", "х", "ц",
                "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я"]

    stop_words = stopwords.words('russian')

    def is_in_ru(self, text):
        for letter in text:
            if letter not in self.alphabet:
                return False
        return True

    def build_tree(self, cur_node, cur_data_node):
        for node in cur_data_node.next_nodes:
            if len(node.next_nodes):
                tmp = NodeClassifier()
                tmp.fit(self.alpha, self.beta, self.max_feat, self.n_est, node, self.n_words)
            else:
                tmp = End(node)
            cur_node.next_nodes.append(tmp)
            self.build_tree(tmp, node)

    def __init__(self, h_tree, alpha=1/2, beta=20, max_feat=3000, n_est=100):
        self.alpha = alpha
        self.beta = beta
        self.max_feat = max_feat
        self.n_est = n_est
        tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n«»0123456789')
        tokenizer.fit_on_texts(h_tree.root.data['entire_text'])

        self.n_words = np.array([word for word
                                in tokenizer.word_counts.keys()
                                if word not in self.stop_words
                                and self.is_in_ru(word)])

        self.root = NodeClassifier()
        self.root.fit(self.alpha, self.beta, self.max_feat, self.n_est, h_tree.root, self.n_words)
        self.build_tree(self.root, h_tree.root)

    def predict(self, node, item):
        next_step = node.predict(item)
        if type(node.next_nodes[next_step]) == End:
            return node.next_nodes[next_step].predict()
        else:
            return self.predict(node.next_nodes[next_step], item)
