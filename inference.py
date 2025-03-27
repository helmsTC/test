import os
import torch
import yaml
import numpy as np
from easydict import EasyDict as edict
from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
from mask_pls.models.mask_model import MaskPS


def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))


def load_model(checkpoint_path, cfg):
    model = MaskPS(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


# Simplified inference for one point cloud from np.ndarray
def infer_single_pointcloud(model, data_module, pointcloud_array):
    # Convert np.ndarray to tensor and format as required by the model
    data = data_module.dataset.preprocess_pointcloud(pointcloud_array)
    with torch.no_grad():
        result = model(data)
    return result


# Main
if __name__ == "__main__":
    checkpoint_path = "path/to/your/checkpoint.ckpt"  # specify your checkpoint path
    
    model_cfg = edict(
        yaml.safe_load(open(os.path.join(getDir(__file__), "../config/model.yaml")))
    )
    backbone_cfg = edict(
        yaml.safe_load(open(os.path.join(getDir(__file__), "../config/backbone.yaml")))
    )
    decoder_cfg = edict(
        yaml.safe_load(open(os.path.join(getDir(__file__), "../config/decoder.yaml")))
    )

    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    cfg.EVALUATE = True

    data_module = SemanticDatasetModule(cfg)

    model = load_model(checkpoint_path, cfg)

    # Example pointcloud array (replace with actual data)
    pointcloud_array = np.random.rand(1000, 4)  # Replace with your actual pointcloud np.ndarray

    # Run inference
    result = infer_single_pointcloud(model, data_module, pointcloud_array)

    # Output result
    print("Inference result:", result)
