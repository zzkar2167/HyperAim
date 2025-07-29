import os.path as osp
import pickle
import numpy as np
import torch

from torch_scatter import scatter_add

from typing import Optional, List, Tuple

def check_mask_exclusivity(train_mask, val_mask, test_mask):
    overlap_train_val = torch.logical_and(train_mask, val_mask).sum().item()
    overlap_train_test = torch.logical_and(train_mask, test_mask).sum().item()
    overlap_val_test = torch.logical_and(val_mask, test_mask).sum().item()
    if overlap_train_val > 0 or overlap_train_test > 0 or overlap_val_test > 0:
        print(
            f"Data leakage detected: train-val overlap={overlap_train_val}, train-test overlap={overlap_train_test}, val-test overlap={overlap_val_test}")
        return False
    return True
class BaseDataset(object):
    def __init__(self, type: str, name: str, device: str = 'cpu', sub_name: Optional[str] = None):
        self.type = type
        self.name = name
        self.device = device
        self.sub_name = sub_name
        if sub_name:
            self.dataset_dir = osp.join('datasets', self.name, self.sub_name)
        else:
            self.dataset_dir = osp.join('datasets', self.name)
        self.split_dir = osp.join(self.dataset_dir, 'splits')
        if not osp.exists(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        self.load_dataset()
        self.preprocess_dataset()
    def load_dataset(self):
        try:
            with open(osp.join(self.dataset_dir, 'features.pickle'), 'rb') as f:
                self.features = pickle.load(f)
            with open(osp.join(self.dataset_dir, 'hypergraph.pickle'), 'rb') as f:
                self.hypergraph = pickle.load(f)
            with open(osp.join(self.dataset_dir, 'labels.pickle'), 'rb') as f:
                self.labels = pickle.load(f)
            # 验证节点数
            num_nodes_features = self.features.shape[0] if isinstance(self.features, np.ndarray) else \
            self.features.shape[0]
            num_nodes_labels = len(self.labels)
            if num_nodes_features != num_nodes_labels:
                raise ValueError(f"Inconsistent node counts: features={num_nodes_features}, labels={num_nodes_labels}")
            print(f"Loaded dataset: {num_nodes_features} nodes, {num_nodes_labels} labels")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Dataset file not found in {self.dataset_dir}: {e}")

    def load_splits(self, split_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        split_file = osp.join(self.split_dir, f'split_{split_idx}.pickle')

        if not osp.exists(split_file):
            raise FileNotFoundError(f"划分文件未找到: {split_file}")

        with open(split_file, 'rb') as f:
            split_data = pickle.load(f)
            if isinstance(split_data, dict) and 'train_mask' in split_data:
                train_mask = split_data['train_mask'].to(self.device)
                val_mask = split_data['val_mask'].to(self.device)
                test_mask = split_data['test_mask'].to(self.device)
            else:
                train_mask, val_mask, test_mask = split_data
                train_mask = train_mask.to(self.device)
                val_mask = val_mask.to(self.device)
                test_mask = test_mask.to(self.device)

            if not check_mask_exclusivity(train_mask, val_mask, test_mask):
                raise ValueError(f"划分文件 {split_file} 无效: 掩码不互斥")
            print(f"从 {split_file} 加载划分 {split_idx}")
            return train_mask, val_mask, test_mask

    def load_all_splits(self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        splits = []
        max_splits = 20
        for i in range(1, max_splits + 1):
            try:
                train_mask, val_mask, test_mask = self.load_splits(i)
                splits.append((train_mask, val_mask, test_mask))
            except FileNotFoundError as e:
                print(f"警告: {e}。跳过划分 {i}。")
                continue

        if not splits:
            raise ValueError(f"在 {self.split_dir} 中未找到有效划分文件")
        print(f"从 {self.split_dir} 加载了 {len(splits)} 个划分")
        return splits

    def preprocess_dataset(self):
        if isinstance(self.hypergraph, dict):
            edge_set = set(self.hypergraph.keys())
            edge_to_num = {}
            num_to_edge = {}
            num = 0
            for edge in edge_set:
                edge_to_num[edge] = num
                num_to_edge[num] = edge
                num += 1
            incidence_matrix = []
            processed_hypergraph = {}
            for edge in edge_set:
                nodes = self.hypergraph[edge]
                processed_hypergraph[edge_to_num[edge]] = nodes
                for node in nodes:
                    incidence_matrix.append([node, edge_to_num[edge]])
            self.processed_hypergraph = processed_hypergraph
            self.hyperedge_index = torch.LongTensor(incidence_matrix).T.contiguous()
            self.edge_to_num = edge_to_num
            self.num_to_edge = num_to_edge
        elif isinstance(self.hypergraph, torch.Tensor):
            print(f"Hypergraph is a torch.Tensor, using it directly as hyperedge_index")
            self.hyperedge_index = self.hypergraph.long().contiguous()
            self.processed_hypergraph = None
            self.edge_to_num = None
            self.num_to_edge = None
        else:
            raise ValueError(f"Unsupported hypergraph type: {type(self.hypergraph)}")

        if isinstance(self.features, torch.Tensor):
            self.features = self.features
        elif isinstance(self.features, np.ndarray):
            self.features = torch.as_tensor(self.features)
        else:
            self.features = torch.as_tensor(self.features.toarray())

        if isinstance(self.labels, np.ndarray):
            if self.labels.dtype == np.object_ or self.labels.dtype.kind in ['U', 'S']:
                unique_labels = np.unique(self.labels)
                label_map = {label: idx for idx, label in enumerate(unique_labels)}
                self.labels = np.array([label_map[label] for label in self.labels], dtype=np.int64)
                print(f"Converted labels: {label_map}")
            else:
                self.labels = np.array(self.labels, dtype=np.int64)
        elif not isinstance(self.labels, torch.Tensor):
            self.labels = np.array(self.labels, dtype=np.int64)
        self.labels = torch.LongTensor(self.labels)

        self.num_nodes = self.features.shape[0]
        self.num_edges = int(self.hyperedge_index[1].max()) + 1 if self.hyperedge_index.numel() > 0 else 0

        weight = torch.ones(self.num_edges)
        Dn = scatter_add(weight[self.hyperedge_index[1]], self.hyperedge_index[0], dim=0, dim_size=self.num_nodes)
        De = scatter_add(torch.ones(self.hyperedge_index.shape[1]), self.hyperedge_index[1], dim=0,
                         dim_size=self.num_edges)

        self.to(self.device)

    def to(self, device: str):
        self.features = self.features.to(device)
        self.hyperedge_index = self.hyperedge_index.to(device)
        self.labels = self.labels.to(device)
        self.device = device
        return self

class GenericDataset(BaseDataset):
    def __init__(self, name: str, sub_name: Optional[str] = None, **kwargs):
        type_id = sub_name if sub_name else name
        super().__init__(type_id, name, sub_name=sub_name, **kwargs)

class actorDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('1', 'actor', **kwargs)

class amazonDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('2', 'amazon', **kwargs)

class pokecDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('3', 'pokec', **kwargs)

class twitchDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('4', 'twitch', **kwargs)

class houseDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('5', 'house', **kwargs)

