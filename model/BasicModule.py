import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path, change_opt=True):
        print(path)
        data = torch.load(path)
        if 'opt' in data:
            if change_opt:
                self.opt.parse(data['opt'], print_=False)
            self.load_state_dict(data['d'])
        else:
            self.load_state_dict(data)
        return self.cuda()

    def save(self, name=None,new=False):
        prefix = 'checkpoints/' + self.model_name + '_' +self.opt.type_+'_'
        if name is None:
            name = time.strftime('%m%d_%H:%M:%S.pth')
        path = prefix + name

        if new:
            data = {'opt':self.opt.state_dict(), 'd':self.state_dict()}
        else:
            data=self.state_dict()

        torch.save(data, path)
        return path

    def get_optimizer(self, lr1, lr2=0, weight_decay=0):
        ignored_params = list(map(id, self.embed.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, self.parameters())
        if lr2 is None: lr2 = lr1*0.5
        optimizer = torch.optim.Adam([
            dict(params=base_params,
                 weight_decay=weight_decay,
                 lr=lr1),
            {'params': self.embed.parameters(), 'lr':lr2}
        ])
        return optimizer



if __name__ == "__main__":
    print(1)
    # test = Test()
    # print(test.model_name)