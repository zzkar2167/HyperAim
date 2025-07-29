from dataset import (
    actorDataset,
    amazonDataset,
    pokecDataset,
    twitchDataset,
)
from dataset import GenericDataset


class DatasetLoader(object):
    def __init__(self):
        self.dataset_map = {
            'actor': ('actor', None),
            'amazon': ('amazon', None),
            'cora_coauthorship': ('coauthorship', 'cora'),
            'citeseer': ('cocitation', 'citeseer'),
            'cora_cocitation': ('cocitation', 'cora'),
            'pubmed_cite': ('pubmed_cite', None),
            'modelnet40': ('ModelNet40', None),
            'pokec': ('pokec', None),
            'twitch': ('twitch', None),
            'house': ('House', None),
            'imdb': ('imdb', None),
            'aminer': ('aminer', None),
        }
        self.specific_dataset_classes = {
            'actor': actorDataset,
            'amazon': amazonDataset,
            'pokec': pokecDataset,
            'twitch': twitchDataset,
        }

    def load(self, dataset_name: str = 'actor', device: str = 'cpu'):
        """Load the specified dataset."""
        dataset_name = dataset_name.lower()
        if dataset_name not in self.dataset_map:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {list(self.dataset_map.keys())}")

        name, sub_name = self.dataset_map[dataset_name]

        if dataset_name in self.specific_dataset_classes:
            dataset_class = self.specific_dataset_classes[dataset_name]
            print(f"Loading dataset '{dataset_name}' using specific class: {dataset_class.__name__}")
            return dataset_class(device=device)
        else:
            print(f"Loading dataset '{dataset_name}' using GenericDataset")
            return GenericDataset(name=name, sub_name=sub_name, device=device)