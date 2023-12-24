# Body_Segmentation_DensePose

This version of densepose is modified such that it does not calculate uv mapping. Only 'fine_segm' and 'coarse_segm' is calculated by the model.

Package tested and working on:  
**Linux-x64**  
**Python v3.11.5**  
**Torch v2.1.2**  
**Torchvision v0.16.2**

**Note: Only works on Linux**

## How to install

```bash
pip install git+https://github.com/VTON-Project/Body_Segmentation_DensePose@main
```

## How to run 'visualize.py'

This script is created to demonstrate how to generate fine segmentation using densepose.

1. Clone the repository and move into the folder.
```bash
git clone https://github.com/VTON-Project/Body_Segmentation_DensePose.git
cd Body_Segmentation_DensePose
```

2. [Install PyTorch](https://pytorch.org/get-started/locally/).

2. Install this densepose package.
```bash
pip install -e .
```

3. Edit input and output image paths in `visualize.py`.

4. Run the `visualize.py`.

***A sample input image and its segmentation is provided in `samples` directory.***