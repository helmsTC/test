"""
maskpls_live.py

Module for real-time inference of MaskPS on single pointcloud frames.
Accepts NumPy arrays or file paths to x,y,z,intensity data and returns per-point semantic
and instance IDs.
"""
import os
import torch
import numpy as np
import yaml
from easydict import EasyDict as edict
from mask_pls.models.mask_model import MaskPS
from mask_pls.datasets.semantic_dataset import get_things_ids


def load_config(config_dir: str) -> edict:
    """
    Load and merge model, backbone, and decoder configs from YAML files.
    """
    def _load(fname):
        path = os.path.join(config_dir, fname)
        return edict(yaml.safe_load(open(path)))

    model_cfg = _load("model.yaml")
    backbone_cfg = _load("backbone.yaml")
    decoder_cfg = _load("decoder.yaml")
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
    cfg.EVALUATE = True
    return cfg


class MaskPSLiveInference:
    def __init__(
        self,
        weights_path: str,
        config_dir: str,
        dataset_name: str = None,
        device: str = "cuda:0",
    ):
        """
        Initialize model and set up thing class IDs for panoptic inference.

        Args:
          weights_path: Path to .pth checkpoint with state_dict.
          config_dir: Directory containing model.yaml, backbone.yaml, decoder.yaml.
          dataset_name: Name of dataset (e.g., 'KITTI' or 'NUSCENES') to fetch thing IDs.
          device: Torch device string.
        """
        # load config and model
        self.cfg = load_config(config_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = MaskPS(self.cfg).to(self.device)
        ckpt = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        # set things_ids and stuff_ids directly without dataset
        from mask_pls.datasets.semantic_dataset import get_things_ids, get_stuff
        things_ids = get_things_ids(dataset_name)
        stuff_ids = list(get_stuff(dataset_name).keys())
        # attach minimal datamodule namespace with both IDs
        from types import SimpleNamespace
        self.model.trainer = SimpleNamespace(
            datamodule=SimpleNamespace(
                things_ids=things_ids,
                stuff_ids=stuff_ids
            )
        )

    def preprocess(
        self,
        cloud: np.ndarray = None,
        file_path: str = None,
    ) -> dict:
        """
        Produce input dict for MaskPS.forward from a pointcloud array or file.

        Args:
          cloud: (N,4) array of x,y,z,intensity
          file_path: local path to .bin file
        Returns:
          A dict with keys matching dataset pipeline: 'pt_coord', 'feats', etc.
        """
        if file_path is not None:
            cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        assert cloud is not None, "Provide cloud array or file_path"

        xyz = torch.from_numpy(cloud[:, :3]).float()
        intensity = torch.from_numpy(cloud[:, 3:4]).float()

        x = {
            'pt_coord': [xyz.to(self.device)],
            'feats': [intensity.to(self.device)],
            'sem_label': [[]],
            'ins_label': [[]],
            'masks': [[]],
            'masks_cls': [[]],
            'masks_ids': [[]],
            'fname': [''],
            'pose': [None],
            'token': [''],
        }
        return x

    def infer(
        self,
        cloud: np.ndarray = None,
        file_path: str = None,
    ) -> (np.ndarray, np.ndarray):
        """
        Run one frame through the model, returning semantic and instance labels.

        Args:
          cloud: (N,4) numpy array of points
          file_path: path to binary file
        Returns:
          sem: (N,) int32 array of semantic labels
          inst: (N,) int32 array of instance IDs
        """
        x = self.preprocess(cloud, file_path)
        with torch.no_grad():
            outputs, padding, _ = self.model(x)
            sem_np, inst_np = self.model.panoptic_inference(outputs, padding)

        return sem_np[0], inst_np[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Real-time MaskPS inference on single pointcloud"
    )
    parser.add_argument(
        "-w", "--weights", required=True, help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "-c", "--config_dir", required=True,
        help="Folder with model.yaml, backbone.yaml, decoder.yaml"
    )
    parser.add_argument(
        "-n", "--dataset_name", default=None,
        help="Dataset name for thing IDs, e.g. 'KITTI' or 'NUSCENES'"
    )
    parser.add_argument(
        "-p", "--pcd", required=True, help="Path to .bin pointcloud file"
    )
    parser.add_argument(
        "-d", "--device", default="cuda:0", help="Torch device"
    )
    args = parser.parse_args()

    app = MaskPSLiveInference(
        args.weights, args.config_dir, args.dataset_name, args.device
    )
    sem, inst = app.infer(file_path=args.pcd)
    print(f"Semantic labels shape: {sem.shape}\nInstance IDs shape: {inst.shape}")
