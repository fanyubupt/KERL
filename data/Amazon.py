from data.Dataset import DataSet

class Beauty(DataSet):
    def __init__(self):
        self.dir_path = './data/dataset/Amazon/Beauty/'
        self.user_record_file = 'Beauty_item_sequences.pkl'
        self.user_mapping_file = 'Beauty_user_mapping.pkl'
        self.item_mapping_file = 'Beauty_item_mapping.pkl'
        self.kg_file = 'embedding.txt'

        self.num_users = 22363
        self.num_items = 12101
        self.vocab_size = 0

        self.user_records = None
        self.user_mapping = None
        self.item_mapping = None

    def generate_dataset(self, index_shift=1):
        user_records = self.load_pickle(self.dir_path + self.user_record_file)
        user_mapping = self.load_pickle(self.dir_path + self.user_mapping_file)
        item_mapping = self.load_pickle(self.dir_path + self.item_mapping_file)
        kg_mapping = self.load_kg(self.dir_path+self.kg_file,self.num_items)
        assert self.num_users == len(user_mapping) and self.num_items == len(item_mapping)

        user_records = self.data_index_shift(user_records, increase_by=index_shift)

        # split dataset
        train_set, test_set = self.split_data_sequentially(user_records, test_radio=0.2)
        # val_num = int(len(test_set)*0.05)
        # val_set_idx = random.sample(range(0, len(test_set)), val_num)
        #
        # val_set =[]
        # new_test_set = []
        # for i in range(0,len(test_set)):
        #     if i in val_set_idx:
        #         val_set.append(test_set[i])
        #     else:
        #         new_test_set.append(test_set[i])
        #
        # return train_set, val_set, new_test_set, self.num_users, self.num_items + index_shift, kg_mapping
        return train_set, test_set, self.num_users, self.num_items + index_shift, kg_mapping