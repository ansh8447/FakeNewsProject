from utils import *
from main import train_and_predict_3_steps
import codecs
import csv
from numba import jit

TRAINBODYCSV = '..//data//train_bodies.csv'
TRAINSTANCECSV = '..//data//train_stances.csv'

TESTBODYCSV = '..//data//test_bodies.csv'
TESTHEADLINECSV = '..//data//test_stances_unlabeled.csv'


print (TESTBODYCSV, TESTHEADLINECSV)

id2body, id2body_sentences = load_body(TRAINBODYCSV)
test_id2body, test_id2body_sentences = load_body(TESTBODYCSV)

for (body_id, body) in test_id2body.iteritems():
    if body_id in id2body and body != id2body[body_id]:
        print ('[Fatal Error] body_id is ambiguous!')
        exit(-1)

id2body.update(test_id2body)
id2body_sentences.update(test_id2body_sentences)

train_data = load_stance(TRAINSTANCECSV)[1:]

seen_head = set()
seen_body_id = set()
for (head, body_id, stance) in train_data:
    seen_head.add(' '.join(head))
    seen_body_id.add(body_id)

test_data = load_title(TESTHEADLINECSV)
print (len(test_data))

overlap = 0
for (head, body_id) in test_data:
    if ' '.join(head) in seen_head or body_id in seen_body_id:
        overlap += 1
print ('overlap =', float(overlap) / len(test_data))

# test_pred, test_scores,train,flag = train_and_predict_3_steps(train_data, test_data, id2body, id2body_sentences)
# print("Hello")



check=0
# print(train)
print("Predictions  Before Online Learning")
test_pred, test_scores, train, flag = train_and_predict_3_steps(train_data, test_data, id2body, id2body_sentences, check)
print (len(test_pred), len(test_scores))

with open('..//resources//Prediction_scores_before.csv', 'w') as out:
    for scores in test_scores:
        out.write(','.join([str(score) for score in scores]) + '\n')


def load_dataset(filename):
    loads = [0,1,2]
    if 1 in loads:
        with open(filename) as fh:
            reader = csv.DictReader(fh)
            data = list(reader)
        return data

test_dataset = load_dataset(TESTHEADLINECSV)

with open('..//resources//Predictions_before.csv', 'w') as csvfile:


    fieldnames = ['Headline', 'Body ID', 'Stance']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i, t in enumerate(test_dataset):
        t['Stance'] = test_pred[i]
        writer.writerow(t)



#Online Learning

check = 1
flag = 1
while(flag == 1):
    # print(train)
    print("Length of the training set",len(train_data))
    test_pred, test_scores, train, flag = train_and_predict_3_steps(train_data, test_data, id2body, id2body_sentences,check)
    train_data =train

print (len(test_pred), len(test_scores))

with open('..//resources//submission_scores_after.csv', 'w') as out:
    for scores in test_scores:
        out.write(','.join([str(score) for score in scores]) + '\n')


def load_dataset(filename):
    loads = [0,1,2]
    if 1 in loads:
        with open(filename) as fh:
            reader = csv.DictReader(fh)
            data = list(reader)
        return data

test_dataset = load_dataset(TESTHEADLINECSV)

with open('..//resources//predictions_after.csv', 'w') as csvfile:


    fieldnames = ['Headline', 'Body ID', 'Stance']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i, t in enumerate(test_dataset):
        t['Stance'] = test_pred[i]
        writer.writerow(t)

