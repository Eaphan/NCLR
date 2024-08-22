import torch
import random

class RandomFlip(object):

    def __init__(self, item_list) -> None:
        self.item_list = item_list

    def __call__(self, data):

        # if torch.randint(0, 2, size=(1,)).item():
        #     for item in self.item_list:
        #         if item not in data:
        #             continue
        #         if len(data[item].shape) == 2:
        #             data[item][:, 0] = -data[item][:, 0]
        #         elif len(data[item].shape) == 1:
        #             data[item][0] = -data[item][0]
        #         else:
        #             raise NotImplementedError

        for item in self.item_list:
            for curr_ax in range(2):
                if random.random() < 0.5:
                    data[item][:, curr_ax] = -data[item][:, curr_ax]

        return data
