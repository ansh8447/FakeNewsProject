import os
import nltk
import numpy as np
import xgboost as xgb
from collections import defaultdict
from math import log
from utils import LABELS
from features import extract_features
from relatedness import train_relatedness_classifier
from gensim.models.keyedvectors import KeyedVectors
from utils import normalize_word
from numba import jit
import random

word2vec = {}


# Load Google's pre-trained Word2Vec model.
def initialize():
    global word2vec
    if len(word2vec) == 0:
        print ('loading word2vec...')
        word_vectors = KeyedVectors.load_word2vec_format('..//resources//GoogleNews-vectors-negative300.bin', binary=True)
        for word in word_vectors.vocab:
            word2vec[normalize_word(word)] = word_vectors[word]
        print ('word2vec loaded')


def prepare_idf(id2body):
    print("in idf(id2body)")
    idf = defaultdict(float)
    for (body_id, body) in id2body.iteritems():
        for word in set(body):
            idf[word] += 1
    for word in idf:
        idf[word] = log(len(id2body) / idf[word])
    return idf

import xgboost as xgb

# def train_further(train, test, id2body, id2body_sentences):
#     idf = prepare_idf(id2body)
#     global word2vec
#     initialize()
#
#     trainX = [extract_features(clean_title, id2body[body_id], id2body_sentences[body_id], idf, word2vec) for
#               (clean_title, body_id, stance) in train]
#     trainY = [int(stance == 'unrelated') for (clean_title, body_id, stance) in train]
#     relatedness_classifier = train_relatedness_classifier(trainX, trainY)
#
#     relatedTrainX = [trainX[i] for i in xrange(len(trainX)) if trainY[i] == 0]
#     relatedTrainY = [int(train[i][2] == 'discuss') for i in xrange(len(trainX)) if trainY[i] == 0]
#     discuss_classifier = train_relatedness_classifier(relatedTrainX, relatedTrainY)
#
#     agreeTrainX = [trainX[i] for i in xrange(len(trainX)) if train[i][2] == 'agree' or train[i][2] == 'disagree']
#     agreeTrainY = [int(train[i][2] == 'agree') for i in xrange(len(trainX)) if
#                    train[i][2] == 'agree' or train[i][2] == 'disagree']
#     agree_classifier = train_relatedness_classifier(agreeTrainX, agreeTrainY)
#     return relatedness_classifier, discuss_classifier, agree_classifier

# def prediction(train, test, id2body, id2body_sentences,relatedness_classifier, discuss_classifier, agree_classifier):
#     idf = prepare_idf(id2body)
#     global word2vec
#     initialize()
#     flag=0
#     title = []
#     bodyNew = []
#     testX = []
#
#     for (clean_title, body_id) in test:
#         Test_X = extract_features(clean_title, id2body[body_id], id2body_sentences[body_id], idf, word2vec)
#         testX.append(Test_X)
#         title.append(clean_title)
#         bodyNew.append(body_id)
#         for i in range(len(testX)):
#             # test[i]
#             # print(title[i])
#             xg_test = xgb.DMatrix(testX[i])
#
#             relatedness_pred = relatedness_classifier.predict(xg_test)
#             discuss_pred = discuss_classifier.predict(xg_test)
#             agree_pred = agree_classifier.predict(xg_test)
#             ret, scores = [], []
#             for (pred_relate, pred_discuss, pred_agree) in zip(relatedness_pred, discuss_pred, agree_pred):
#                 scores.append((pred_relate, pred_discuss, pred_agree))
#                 if pred_relate >= 0.5:
#                     ret.append('unrelated')
#                     flag =0
#                 elif pred_discuss >= 0.5:
#                     print(pred_discuss)
#                     print(title[i])
#                     print(bodyNew[i])
#                     decision = input("Is it a fake news(0 for yes/1 for no):")
#                     if decision == 0:
#                         stanceNew = 'disagree'
#                         train.append((title[i], bodyNew[i], stanceNew))
#
#                         flag=1
#                     else:
#                         stanceNew = 'agree'
#                         train.append((title[i], bodyNew[i], stanceNew))
#                         flag=1
#                 elif pred_agree >= 0.5:
#                     ret.append('agree')
#                     flag = 0
#                 else:
#                     ret.append('disagree')
#                     flag = 0
#     return ret, scores,train,flag


def train_and_predict_3_steps(train, test, id2body, id2body_sentences, check):
    print("in_train_and_predict")
    # flag = 0
    idf = prepare_idf(id2body)
    global word2vec
    initialize()
    discuss_count = 0

    trainX = [extract_features(clean_title, id2body[body_id], id2body_sentences[body_id],idf, word2vec) for (clean_title, body_id, stance) in train]
    trainY = [int(stance == 'unrelated') for (clean_title, body_id, stance) in train]
    relatedness_classifier = train_relatedness_classifier(trainX, trainY)
    print("training done")
    relatedTrainX = [trainX[i] for i in xrange(len(trainX)) if trainY[i] == 0]
    relatedTrainY = [int(train[i][2] == 'discuss') for i in xrange(len(trainX)) if trainY[i] == 0]
    discuss_classifier = train_relatedness_classifier(relatedTrainX, relatedTrainY)
    print("training done")
    agreeTrainX = [trainX[i] for i in xrange(len(trainX)) if train[i][2] == 'agree' or train[i][2] == 'disagree']
    agreeTrainY = [int(train[i][2] == 'agree') for i in xrange(len(trainX)) if train[i][2] == 'agree' or train[i][2] == 'disagree']
    agree_classifier = train_relatedness_classifier(agreeTrainX, agreeTrainY)
    print("training done")
    # relatedness_classifier, discuss_classifier, agree_classifier = train_further(train, test, id2body, id2body_sentences)

    # testX = [extract_features(clean_title, id2body[body_id], id2body_sentences[body_id],idf, word2vec) for (clean_title, body_id) in test]
    title = []
    bodyNew = []
    testX = []

    for (clean_title, body_id) in test :
        Test_X = extract_features(clean_title, id2body[body_id], id2body_sentences[body_id], idf, word2vec)
        testX.append(Test_X)
        title.append(clean_title)
        bodyNew.append(body_id)

    # print(title[1])
    # print(testX[1])
    # print(bodyNew[1])

    # ret,scores,train,flag = prediction(train, test, id2body, id2body_sentences,relatedness_classifier, discuss_classifier, agree_classifier)
    #     ret, scores = [], []
        #flag=0
        print("extraction  done")
        if check == 0:
            print("in check zero")
            ret, scores = [], []
            #for i in range(len(testX)):
                # test[i]
                # print(title[i])
            xg_test = xgb.DMatrix(testX)

            relatedness_pred = relatedness_classifier.predict(xg_test);
            discuss_pred = discuss_classifier.predict(xg_test)
            agree_pred = agree_classifier.predict(xg_test)

            for (pred_relate, pred_discuss, pred_agree) in zip(relatedness_pred, discuss_pred, agree_pred):
                scores.append((pred_relate, pred_discuss, pred_agree))
                if pred_relate >= 0.5:
                    ret.append('unrelated')
                    # scores.append((pred_relate, pred_discuss, pred_agree))
                elif pred_discuss >= 0.5:
                    ret.append('discuss')
                    discuss_count = discuss_count + 1

                elif pred_agree >= 0.5:
                    ret.append('agree')
                    # scores.append((pred_relate, pred_discuss, pred_agree))

                # elif pred_agree >= 0.5:
                #     ret.append('agree')
                #     scores.append((pred_relate, pred_discuss, pred_agree))
                else:
                    ret.append('disagree')
                    scores.append((pred_relate, pred_discuss, pred_agree))

        else:
            print("in check else")
            ret, scores = [], []
            for i in range(len(testX)):
                # test[i]
                # print(title[i])
                xg_test = xgb.DMatrix(testX[i])

                relatedness_pred = relatedness_classifier.predict(xg_test);
                discuss_pred = discuss_classifier.predict(xg_test)
                agree_pred = agree_classifier.predict(xg_test)
                for (pred_relate, pred_discuss, pred_agree) in zip(relatedness_pred, discuss_pred, agree_pred):
                    scores.append((pred_relate, pred_discuss, pred_agree))
                    if pred_relate >= 0.5:
                        ret.append('unrelated')
                        # scores.append((pred_relate, pred_discuss, pred_agree))
                    elif pred_discuss >= 0.5:
                        if i < 100:
                            print(title[i])
                            print(bodyNew[i])
                            #decision = input("Is it a fake news(0 for yes/1 for no):")
                            decision = random.randint(0, 1)
                            if decision == 0:
                                stanceNew = 'disagree'
                                for a in range(0, 50):
                                    train.append((title[i], bodyNew[i], stanceNew))
                                flag = 1

                                ret.append('disagree')

                            else:
                                stanceNew = 'agree'
                                for a in range(0, 50):
                                    train.append((title[i], bodyNew[i], stanceNew))
                                flag=1
                                ret.append('agree')
                            return ret, scores, train, flag
                        else:
                            print(title[i])
                            print(bodyNew[i])
                            ret.append('discuss')
                            discuss_count = discuss_count + 1

                    elif pred_agree >= 0.5:
                        ret.append('agree')
                        # scores.append((pred_relate, pred_discuss, pred_agree))

                    # elif pred_agree >= 0.5:
                    #     ret.append('agree')
                    #     scores.append((pred_relate, pred_discuss, pred_agree))
                    else:
                        ret.append('disagree')
                        scores.append((pred_relate, pred_discuss, pred_agree))
    flag = 0
    print("Total Discuss count: ", discuss_count)
    return ret, scores, train, flag

