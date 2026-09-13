# Plant Disease Semantic Segmentation

A lightweight attention-gated U-Net baseline for separating symptomatic leaf regions
from background. It is designed as a reproducible computer-vision starting point for
PlantSeg-style experiments and uses a synthetic mask in the demo.

## What it demonstrates

- Encoder-decoder segmentation with skip connections
- Attention gates that filter encoder features before fusion
- Binary logits suitable for Dice/BCE training
- A dataset-agnostic place to add augmentation and threshold calibration

## Run

~~~bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m src.demo
~~~

The demo is not a reproduction of a published benchmark and reports no new result.
Use an independently annotated test split before comparing mIoU or F1.

## Structure

- src/model.py — attention-gated segmentation network
- src/demo.py — synthetic leaf-mask smoke test

## License

Apache-2.0
