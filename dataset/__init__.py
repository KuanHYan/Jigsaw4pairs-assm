from .all_piece_matching_dataset import build_all_piece_matching_dataloader
from .dataset_config import dataset_cfg
from .custom_dataset import build_custom_dataset


def build_dataloader(cfg):
    dataset = cfg.DATASET.lower().split(".")
    if dataset[0] == "breaking_bad":
        if dataset[1] == "all_piece_matching":
            return build_all_piece_matching_dataloader(cfg)
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented")
    elif dataset[0] == "custom":
        return build_custom_dataset(cfg)
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")
