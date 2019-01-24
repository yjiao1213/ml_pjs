import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import sklearn
import os
from collections import Counter


def make_word_dict(dirname):
    '''
    :param dirname: 存放邮件的目录
    :return: 取出最高频的5000个words，[(word, times),...]
    '''
    mails = [os.path.join(dirname,f) for f in os.listdir(dirname)]
    all_words = []
    for mail in mails:
        with open(mail) as m:
            for i, line in enumerate(m):
                if i == 2:
                    words = line.split()
                    all_words += words
    dic_words = Counter(all_words)

    keys = list(dic_words.keys())
    for key in keys:
        if key.isalpha() == False:
            del dic_words[key]
        elif len(key) <= 1:
            del dic_words[key]
        else:
            pass
    dic_words = dic_words.most_common(5000)
    return dic_words


def extract_feature(dirname, dic_words):
    '''
    :param dirname: 存放邮件的目录
    :param dic_words: 高频词字典
    :return:
    '''
    mails = [os.path.join(dirname, f) for f in os.listdir(dirname)]
    feature_matrix = np.zeros((len(mails), 5000))
    docID = 0;
    for mail in mails:
        with open(mail) as m:
            for i, line in enumerate(m):
                if i == 2:
                    words = line.split()
                    for word in words:
                        # wordID = 0
                        for i,d in enumerate(dic_words):
                            if d[0] == word:
                                wordID = i
                                feature_matrix[docID, wordID] = words.count(word)
        docID += 1
    return feature_matrix




if __name__ == "__main__":
    train_file = os.path.join(os.getcwd(),"data", "train-mails")
    test_file = os.path.join(os.getcwd(), "data", "test-mails")

    dic_train = make_word_dict(train_file)

    feature_matrix_train = extract_feature(train_file, dic_train)
    test_matrix = extract_feature(test_file, dic_train)

    train_labels = np.zeros(702)
    train_labels[351:701] = 1

    test_labels = np.zeros(260)
    test_labels[130:260] = 1

    # 使用 SVM 和 Naive bayes
    model1 = LinearSVC()
    model2 = MultinomialNB()

    model1.fit(feature_matrix_train, train_labels)
    model2.fit(feature_matrix_train, train_labels)

    # 测试数据
    predict1 = model1.predict(test_matrix)
    predict2 = model2.predict(test_matrix)

    print(sklearn.metrics.confusion_matrix(test_labels, predict1))
    print(sklearn.metrics.confusion_matrix(test_labels, predict2))
