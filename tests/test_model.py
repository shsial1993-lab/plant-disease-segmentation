import torch

from src.model import PlantSegmentationNet


def test_segmentation_preserves_spatial_size() -> None:
    model = PlantSegmentationNet()
    logits = model(torch.randn(1, 3, 64, 64))
    assert logits.shape == (1, 1, 64, 64)
