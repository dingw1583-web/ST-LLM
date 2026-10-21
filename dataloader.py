import dgl
import torch
from dgl.data import DGLDataset

from config import Config
import numpy as np


config = Config()


class MixTrafficFlowDataset4DGL(DGLDataset):
    def __init__(self, payload_path, header_path):
        self.payload_path = payload_path
        self.header_path = header_path

        super(MixTrafficFlowDataset4DGL, self).__init__(name="MixTrafficFlowDataset4DGL")


    def process(self):

        self.payload_data, self.payload_labels = dgl.load_graphs(self.payload_path)


        header_file = np.load(self.header_path, allow_pickle=True)
        self.header_data = header_file['data']
        self.header_labels = header_file['label']


        self.labels = self.payload_labels["glabel"]


        factor = len(self.payload_data) // len(self.header_data)
        print(f"复制因子: payload({len(self.payload_data)})/header({len(self.header_data)}) = {factor}")


        if len(self.payload_data) % len(self.header_data) != 0:
            raise ValueError(
                f"Payload数量({len(self.payload_labels)})不是header数量({len(self.header_labels)})的整数倍"
            )


        unique_labels, counts = np.unique(self.labels, return_counts=True)


        self.num_samples = len(self.payload_data) // config.FLOW_PAD_TRUNC_LENGTH


        self.num_samples = len(self.header_data)


        print(f"Dataset loaded successfully: {self.num_samples} samples")


    def __getitem__(self, index):
        start_ind = config.FLOW_PAD_TRUNC_LENGTH * index
        end_ind = start_ind + config.FLOW_PAD_TRUNC_LENGTH


        header_seq = self.header_data[index]


        header_tensor = torch.tensor(header_seq, dtype=torch.long)


        return header_tensor, self.payload_data[start_ind: end_ind], self.labels[index]


    def __len__(self):


        return len(self.header_labels)

if __name__ == '__main__':
    pass
