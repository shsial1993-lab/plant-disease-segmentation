from __future__ import annotations

import torch

from .model import PlantSegmentationNet


def main() -> None:
    torch.manual_seed(7)
    model = PlantSegmentationNet()
    image = torch.rand(2, 3, 128, 128)
    logits = model(image)
    print('image:', tuple(image.shape), 'mask logits:', tuple(logits.shape))


if __name__ == '__main__':
    main()
