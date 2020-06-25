# KERL Model for Sequential Recommendation
The implementation of the paper:

*Pengfei Wang, Yu Fan, Long Xia, Wayne Xin Zhao, Shaozhang Niu and Jimmy Huang
, "**KERL: A Knowledge-Guided Reinforcement Learning Model for Sequential Recommendation**", **SIGIR 2020*** 


**Please cite our paper if you use our code. Thanks!**

Author: Yu Fan (fanyubupt@gmail.com)


## Environments

- python 3.6
- PyTorch (version: 1.0.0)
- numpy (version: 1.15.0)
- scipy (version: 1.1.0)
- sklearn (version: 0.19.1)


## Dataset

In our experiments, the *Amazon-Beauty* *Amazon-CDs* and *Amazon-Books* datasets are from http://jmcauley.ucsd.edu/data/amazon/, the *LastFM*  dataset is from http://www.cp.jku.at/datasets/LFM-1b/. 

The ```XXX_tem_sequences.pkl``` file is a list of lists that stores the inner item id of each user in a chronological order, e.g., ```user_records[0]=[item_id0, item_id1, item_id2,...]```.

The ```XXX_user_mapping.pkl``` file is a list that maps the user inner id to its original id, e.g., ```user_mapping[0]=A2SUAM1J3GNN3B```.

The ```XXX_item_mapping.pkl``` file is similar to ```XXX_user_mapping.pkl```.
The beauty dataset is in the file as an example.

### Knowledge Graph Embedding
The KB embedding is trained by transE based on projects of THUNLP(https://github.com/thunlp). OpenKE(https://github.com/thunlp/OpenKE) is their main project, 
you can find nearly all methods related here, including all TransX model used in the paper.

## Example to run the code

Reproducing the results reported in our paper, please run the code as follows:

```
python runRL.py
```

