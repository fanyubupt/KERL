from interactions import Interactions
from eval_metrics import *

import argparse
import logging
from time import time
import datetime
import torch
from model.KERL import kerl
import torch.nn.functional as F
import random
import pickle
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

def generate_testsample(test_set,itemnum):
    '''
    input
        test_set: ground-truth
        itemnum: item number
    output
        all_sample:randomly sampled 100 negative items and 1 positive items
    '''

    all_sample =[]
    for eachset in test_set:
        testsample = []
        for i in range(1):
            onesample = []
            onesample +=[eachset[i]]
            other = list(range(1, itemnum))
            other.remove(eachset[i])
            neg = random.sample(other,100)
            onesample +=neg
            testsample.append(onesample)
        testsample = np.stack(testsample)
        all_sample.append(testsample)
    all_sample = np.stack(all_sample)
    return all_sample

def evaluation_kerl(kerl, train, test_set):
    num_users = train.num_users
    num_items = train.num_items
    batch_size = 1024
    num_batches = int(num_users / batch_size) + 1
    user_indexes = np.arange(num_users)
    item_indexes = np.arange(num_items)
    pred_list = None
    test_sequences = train.test_sequences.sequences
    test_len = train.test_sequences.length

    all_sample = generate_testsample(test_set,num_items)
    for batchID in range(num_batches):
        start = batchID * batch_size
        end = start + batch_size

        if batchID == num_batches - 1:
            if start < num_users:
                end = num_users
            else:
                break

        batch_user_index = user_indexes[start:end]

        batch_test_sequences = test_sequences[batch_user_index]
        batch_test_sequences = np.atleast_2d(batch_test_sequences)
        batch_test_len = test_len[batch_user_index]

        batch_test_len = torch.from_numpy(batch_test_len).type(torch.LongTensor).to(device)
        batch_test_sequences = torch.from_numpy(batch_test_sequences).type(torch.LongTensor).to(device)

        prediction_score = kerl(batch_test_sequences, batch_test_len)
        rating_pred = prediction_score
        rating_pred = rating_pred.cpu().data.numpy().copy()

        if batchID == 0:
            pred_list = rating_pred
        else:
            pred_list = np.append(pred_list, rating_pred, axis=0)

    #rank 101 scores and select top 10,and find the corresponding 10 item indexes
    all_top10 = []
    for i in range(1):
        oneloc_top10 = []
        user_index = 0
        for each_policy,each_s in zip(pred_list[:, i, :],all_sample[:,i,:]):
            #items-->101score
            each_sample = -each_policy[each_s]
            top10index = np.argsort(each_sample)[:10]
            top10item = each_s[top10index]
            oneloc_top10.append(top10item)
        oneloc_top10=np.stack(oneloc_top10)
        all_top10.append(oneloc_top10)
        user_index +=1
    all_top10 = np.stack(all_top10,axis=1)
    pred_list = all_top10

    precision, ndcg = [], []
    k=10
    for i in range(1):
        pred = pred_list[:,i,:]
        precision.append(precision_at_k(test_set, pred, k,i))
        ndcg.append(ndcg_k(test_set, pred, k,i))

    #save results
    # def save_obj(obj, name):
    #     with open(name + '.pkl', 'wb') as f:
    #         pickle.dump(obj, f)
    # str_name = "./result/LFM"+str(precision[0])
    # save_obj(pred_list, str_name)
    return precision, ndcg

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

def train_kerl(train_data, test_data, config,kg_map):
    num_users = train_data.num_users
    num_items = train_data.num_items

    # convert to sequences, targets and users
    sequences_np = train_data.sequences.sequences
    targets_np = train_data.sequences.targets
    users_np = train_data.sequences.user_ids
    trainlen_np = train_data.sequences.length
    tarlen_np = train_data.sequences.tarlen

    n_train = sequences_np.shape[0]
    logger.info("Total training records:{}".format(n_train))


    kg_map = torch.from_numpy(kg_map).type(torch.FloatTensor).to(device)
    kg_map.requires_grad=False
    seq_model = kerl(num_users, num_items, config, device, kg_map).to(device)
    optimizer = torch.optim.Adam(seq_model.parameters(), lr=config.learning_rate,weight_decay=config.l2)

    lamda = 5  #loss function hyperparameter
    print("loss lamda=",lamda)
    CEloss = torch.nn.CrossEntropyLoss()
    margin = 0.0
    MRLoss = torch.nn.MarginRankingLoss(margin=margin)

    record_indexes = np.arange(n_train)
    batch_size = config.batch_size
    num_batches = int(n_train / batch_size) + 1

    stopping_step = 0
    cur_best_pre_0 = 0
    should_stop = False
    for epoch_num in range(config.n_iter):
        t1 = time()
        loss=0
        # set model to training mode
        seq_model.train()

        np.random.shuffle(record_indexes)
        epoch_reward=0.0
        epoch_loss = 0.0
        for batchID in range(num_batches):
            start = batchID * batch_size
            end = start + batch_size

            if batchID == num_batches - 1:
                if start < n_train:
                    end = n_train
                else:
                    break

            batch_record_index = record_indexes[start:end]

            batch_users = users_np[batch_record_index]
            batch_sequences = sequences_np[batch_record_index]
            batch_targets = targets_np[batch_record_index]
            trainlen = trainlen_np[batch_record_index]
            tarlen = tarlen_np[batch_record_index]

            tarlen = torch.from_numpy(tarlen).type(torch.LongTensor).to(device)
            trainlen = torch.from_numpy(trainlen).type(torch.LongTensor).to(device)
            batch_users = torch.from_numpy(batch_users).type(torch.LongTensor).to(device)
            batch_sequences = torch.from_numpy(batch_sequences).type(torch.LongTensor).to(device)
            batch_targets = torch.from_numpy(batch_targets).type(torch.LongTensor).to(device)

            items_to_predict = batch_targets

            if epoch_num>=0:
                pred_one_hot = np.zeros((len(batch_users),num_items))
                batch_tar=targets_np[batch_record_index]
                for i,tar in enumerate(batch_tar):
                    pred_one_hot[i][tar]=0.2/config.T
                pred_one_hot = torch.from_numpy(pred_one_hot).type(torch.FloatTensor).to(device)

                prediction_score,orgin,batch_targets,Reward,dist_sort = seq_model.RLtrain(batch_sequences,
                items_to_predict,pred_one_hot,trainlen,tarlen)

                target = torch.ones((len(prediction_score))).to(device)

                min_reward = dist_sort[0,:].unsqueeze(1)
                max_reward = dist_sort[-1,:].unsqueeze(1)
                mrloss = MRLoss(max_reward,min_reward,target)

                orgin = orgin.view(prediction_score.shape[0] * prediction_score.shape[1], -1)
                target = batch_targets.view(batch_targets.shape[0]*batch_targets.shape[1])
                reward = Reward.view(Reward.shape[0]*Reward.shape[1]).to(device)

                prob = torch.index_select(orgin,1,target)
                prob = torch.diagonal(prob,0)
                RLloss =-torch.mean(torch.mul(reward,torch.log(prob)))
                loss = RLloss+lamda*mrloss
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        epoch_loss /= num_batches
        t2 = time()
        output_str = "Epoch %d [%.1f s]  loss=%.4f" % (epoch_num + 1, t2 - t1, epoch_loss)
        logger.info(output_str)

        if (epoch_num + 1) > 1:
            seq_model.eval()
            precision, ndcg = evaluation_kerl(seq_model, train_data, test_data)
            #precision, ndcg = evaluation_kerl(seq_model, train_data, val_set)
            logger.info(', '.join(str(e) for e in precision))
            logger.info(', '.join(str(e) for e in ndcg))
            logger.info("Evaluation time:{}".format(time() - t2))
            cur_best_pre_0, stopping_step, should_stop = early_stopping(precision[0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc',
                                                                    flag_step=5)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            if should_stop == True:
                break
    logger.info("\n")
    logger.info("\n")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data arguments
    #L: max sequence length
    #T: episode length
    parser.add_argument('--L', type=int, default=50)
    parser.add_argument('--T', type=int, default=3)

    # train arguments
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=0)

    # model dependent arguments
    parser.add_argument('--d', type=int, default=50)

    config = parser.parse_args()

    from data import Amazon
    data_set = Amazon.Beauty()  # Books, CDs, LastFM
    train_set, test_set, num_users, num_items,kg_map = data_set.generate_dataset(index_shift=1)

    maxlen = 0
    for inter in train_set:
        if len(inter)>maxlen:
            maxlen=len(inter)

    train = Interactions(train_set, num_users, num_items)
    train.to_newsequence(config.L, config.T)

    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(config)

    train_kerl(train,test_set,config,kg_map)
