import dgl
import torch
from dgl.data import DGLDataset

import numpy as np



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


        if self.header_data.ndim != 3 or len(self.header_data) == 0:
            raise ValueError("Expected nonempty header arrays with shape [flows, packets, bytes]")
        self.flow_length = self.header_data.shape[1]
        if len(self.payload_data) != len(self.header_data) * self.flow_length:
            raise ValueError("Header and payload packet counts differ; regenerate both modalities with the same configuration")
        if self.labels.tolist() != self.header_labels.tolist():
            raise ValueError("Header and payload labels differ; regenerate aligned data")

        factor = len(self.payload_data) // len(self.header_data)
        print(f"Replication factor: payload({len(self.payload_data)})/header({len(self.header_data)}) = {factor}")


        self.num_samples = len(self.header_data)


        print(f"Dataset loaded successfully: {self.num_samples} samples")


    def __getitem__(self, index):
        start_ind = self.flow_length * index
        end_ind = start_ind + self.flow_length


        header_seq = self.header_data[index]


        header_tensor = torch.tensor(header_seq, dtype=torch.long)


        return header_tensor, self.payload_data[start_ind: end_ind], self.labels[index]


    def __len__(self):


        return len(self.header_labels)

if __name__ == '__main__':
    pass
