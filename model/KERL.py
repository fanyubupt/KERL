import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from eval_metrics import *
from model.DynamicGRU import DynamicGRU
class kerl(nn.Module):
    def __init__(self, num_users, num_items, model_args, device,kg_map):
        super(kerl, self).__init__()

        self.args = model_args
        self.device = device
        self.lamda = 10
        # init args
        L = self.args.L
        dims = self.args.d
        predict_T=self.args.T
        # user and item embeddings
        self.kg_map =kg_map
        # self.user_embeddings = nn.Embedding(num_users, dims).to(device)
        self.item_embeddings = nn.Embedding(num_items, dims).to(device)
        self.DP = nn.Dropout(0.5)
        self.enc = DynamicGRU(input_dim=dims,
                              output_dim=dims, bidirectional=False, batch_first=True)


        self.mlp = nn.Linear(dims+50*2, dims*2)
        self.fc = nn.Linear(dims*2, num_items)
        self.mlp_history = nn.Linear(50,50)
        self.BN = nn.BatchNorm1d(50, affine=False)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, batch_sequences, train_len):
        #test process
        probs = []
        input = self.item_embeddings(batch_sequences)
        out_enc, h = self.enc(input, train_len)

        kg_map = self.BN(self.kg_map)
        kg_map =kg_map.detach()
        batch_kg = self.get_kg(batch_sequences,train_len,kg_map)

        mlp_in = torch.cat([h.squeeze(),batch_kg, self.mlp_history(batch_kg)],dim=1)
        mlp_hidden = self.mlp(mlp_in)
        mlp_hidden = torch.tanh(mlp_hidden)

        out = self.fc(mlp_hidden)
        probs.append(out)
        return torch.stack(probs, dim=1)

    def RLtrain(self, batch_sequences, items_to_predict, pred_one_hot, train_len,tarlen):
        probs = []
        probs_orgin = []
        each_sample = [] 
        Rewards = []
        input = self.item_embeddings(batch_sequences)

        out_enc, h = self.enc(input, train_len)

        kg_map = self.BN(self.kg_map)
        batch_kg = self.get_kg(batch_sequences,train_len,kg_map)
        #state[sequence,history_kg,future_kg]
        mlp_in = torch.cat([h.squeeze(),batch_kg,self.mlp_history(batch_kg)],dim=1)
        mlp_hidden = self.mlp(mlp_in)
        mlp_hidden = torch.tanh(mlp_hidden)
        out_fc = self.fc(mlp_hidden)

        '''
        When sampling episodes, we increased the probability of ground truth to improve the convergence efficiency
        '''
        out_distribution = F.softmax(out_fc, dim=1)
        probs_orgin.append(out_distribution)
        out_distribution = 0.8 * out_distribution
        out_distribution = torch.add(out_distribution, pred_one_hot)
        # pai-->p(a|s)
        probs.append(out_distribution)
        m = torch.distributions.categorical.Categorical(out_distribution)
        # action
        sample1 = m.sample()
        each_sample.append(sample1)
        # generate 3 episode
        Reward, dist_sort = self.generateReward(sample1, self.args.T-1, 3, items_to_predict, pred_one_hot, h,batch_kg,kg_map,tarlen)
        Rewards.append(Reward)
        # dec_input_target = self.item_embeddings(items_to_predict)

        probs = torch.stack(probs, dim=1)
        probs_orgin = torch.stack(probs_orgin, dim=1)
        return probs, probs_orgin, torch.stack(each_sample, dim=1), torch.stack(Rewards, dim=1),dist_sort

    def get_kg(self,batch_sequences,trainlen,kg_map):
        # batch_kg_avg
        batch_kg = []
        for i, seq in enumerate(batch_sequences):
            seq_kg = kg_map[seq]
            seq_kg_avg = torch.sum(seq_kg,dim=0)
            seq_kg_avg = torch.div(seq_kg_avg,trainlen[i])
            batch_kg.append(seq_kg_avg)
        batch_kg = torch.stack(batch_kg)
        return batch_kg

    def generateReward(self, sample1, path_len, path_num, items_to_predict, pred_one_hot,h_orin,batch_kg,kg_map,tarlen):
        history_kg = self.mlp_history(batch_kg)
        Reward = []
        dist = []
        dist_replay = []
        for paths in range(path_num):
            h = h_orin
            indexes = []
            indexes.append(sample1)
            dec_inp_index = sample1
            dec_inp = self.item_embeddings(dec_inp_index)
            dec_inp = dec_inp.unsqueeze(1)
            ground_kg = self.get_kg(items_to_predict[:, self.args.T - path_len - 1:],tarlen,kg_map)
            for i in range(path_len):
                out_enc, h = self.enc(dec_inp, h, one=True)
                # out_fc = self.fc(h.squeeze())
                mlp_in = torch.cat([h.squeeze(), batch_kg, self.mlp_history(batch_kg)], dim=1)
                mlp_hidden = self.mlp(mlp_in)
                mlp_hidden = torch.tanh(mlp_hidden)
                out_fc = self.fc(mlp_hidden)

                out_distribution = F.softmax(out_fc, dim=1)
                out_distribution = 0.8 * out_distribution
                out_distribution = torch.add(out_distribution, pred_one_hot)
                # pai-->p(a|s)
                m = torch.distributions.categorical.Categorical(out_distribution)
                sample2 = m.sample()
                dec_inp = self.item_embeddings(sample2)
                dec_inp = dec_inp.unsqueeze(1)
                indexes.append(sample2)
            indexes = torch.stack(indexes, dim=1)
            episode_kg = self.get_kg(indexes,torch.Tensor([path_len+1]*len(indexes)),kg_map)

            '''
            dist: knowledge reward
            dist_replay: induction network training (rank)
            '''
            dist.append(self.cos(episode_kg ,ground_kg))
            dist_replay.append(self.cos(episode_kg,history_kg))
            #Reward.append(bleu_each(indexes,items_to_predict[:,self.args.T-path_len-1:]))
            Reward.append(dcg_k(items_to_predict[:, self.args.T - path_len - 1:], indexes, path_len + 1))
        Reward = torch.FloatTensor(Reward).to(self.device)
        dist = torch.stack(dist, dim=0)
        dist = torch.mean(dist, dim=0)

        dist_replay = torch.stack(dist_replay, dim=0)
        dist_sort = self.compare_kgReawrd(Reward, dist_replay)
        Reward = torch.mean(Reward, dim=0)
        Reward = Reward + self.lamda * dist
        dist.sort =dist_sort.detach()
        return Reward, dist_sort


    def compare_kgReawrd(self, reward, dist):
        logit_reward, indice = reward.sort(dim=0)
        dist_sort = dist.gather(dim=0, index=indice)
        return dist_sort

