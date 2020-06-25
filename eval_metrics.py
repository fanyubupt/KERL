import math
import numpy as np
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
def bleu(hyps, refs):
    """
    bleu
    """
    bleu_4 = []
    for hyp, ref in zip(hyps, refs):
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[0.5, 0.5, 0, 0])
        except:
            score = 0
        bleu_4.append(score)
    bleu_4 = np.average(bleu_4)
    return bleu_4

def bleu_each(hyps, refs):
    """
    bleu
    """
    bleu_4 = []
    hyps=hyps.cpu().numpy()
    refs=refs.cpu().numpy()
    for hyp, ref in zip(hyps, refs):
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[0.5, 0.5, 0, 0])
        except:
            score = 0
        bleu_4.append(score)
    return bleu_4

def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)
'''
def precision_at_k(actual, predicted, topk,item_i):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = actual[i][item_i]
        pred_set = predicted[i]
        if  act_set in pred_set:
            sum_precision += 1
    print(sum_precision)
    return sum_precision / num_users
'''

def precision_at_k(actual, predicted, topk,item_i):
    sum_precision = 0.0
    user = 0
    num_users = len(predicted)
    for i in range(num_users):
        if actual[i][item_i]>0:
            user +=1
            act_set = actual[i][item_i]
            pred_set = predicted[i]
            if act_set in pred_set:
                sum_precision += 1
        else:
            continue
    #print(user)
    return sum_precision / user

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

'''
def ndcg_k(actual, predicted, topk,item_i):
    k = min(topk, len(actual))
    idcg = idcg_k(k)
    res = 0
    for user_id in range(len(actual)):
        dcg_k = sum([int(predicted[user_id][j] in [actual[user_id][item_i]]) / math.log(j+2, 2) for j in range(k)])
        res += dcg_k
    return res/ float(len(predicted))
'''
def ndcg_k(actual, predicted, topk,item_i):
    k = min(topk, len(actual))
    idcg = idcg_k(k)
    res = 0
    user = 0
    for user_id in range(len(actual)):
        if actual[user_id][item_i] > 0:
            user +=1
            dcg_k = sum([int(predicted[user_id][j] in [actual[user_id][item_i]]) / math.log(j+2, 2) for j in range(k)])
            res += dcg_k
        else:
            continue
    #print(user)
    return res/user

def dcg_k(actual, predicted, topk):
    k = min(topk, len(actual))
    dcgs=[]
    actual = actual.cpu().numpy()
    predicted = predicted.cpu().numpy()
    for user_id in range(len(actual)):
        value = []
        for i in predicted[user_id]:
            try:
                value += [topk -int(np.argwhere(actual[user_id]==i))]
                #print(value)
            except:
                value += [0]
        #dcg_k = sum([int(predicted[user_id][j] in set(actual[user_id])) / math.log(j+2, 2) for j in range(k)])
        dcg_k = sum([value[j] / math.log(j+2, 2) for j in range(k)])
        if dcg_k==0:
           dcg_k=1e-5
        dcgs.append(dcg_k)
    return dcgs
# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res
if __name__ == '__main__':
    actual = [[1, 2], [3, 4, 5]]
    predicted = [[10, 20, 1, 30, 40], [10, 3, 20, 4, 5]]
    print(ndcg_k(actual, predicted, 5))
