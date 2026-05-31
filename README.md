# PoseInvariantShapeEncoding
MSc Thesis - Learned shape encoding with pose normalization for 2D binary shapes using SIREN-based DeepSDF.
# Thesis Code: Bee Wing Shape Alignment and Matching

This repository contains the code used for the computational part of my master's thesis on learning and matching 2D bee wing shapes using signed distance functions, SIREN/DeepSDF models, and similarity transformations.

The code is split into two main notebooks:

- `Phase_1_code_Thesis.ipynb` — preprocessing, alignment, SDF construction, latent-space training.
- `Phase_2_code_Thesis.ipynb` — loading the trained model and performing shape matching/reconstruction from an observed wing image.

---

## Project overview

The goal of the project is to represent wing shapes independently of their pose.  
Each wing image is first normalized with respect to translation, rotation, and scale.  
After this alignment step, each normalized wing is converted into a signed distance function (SDF).  
A SIREN-based DeepSDF model is then trained to represent the family of wing shapes using a learned latent vector for each training sample.

In the second phase, a new observed wing image, possibly incomplete or damaged, is matched by optimizing:

1. a latent code representing the intrinsic wing shape, and  
2. a similarity transformation \(g \in \mathrm{Sim}(2)\) representing pose.

---

## Repository structure

```text
.
├── Phase_1_code_Thesis.ipynb
├── Phase_2_code_Thesis.ipynb
├── my_siren_model.py              # helper file for loading the trained model
├── test1/
│   ├── mask/                      # binary wing masks used for training
│   └── real/                      # optional real/reference images
├── model_files/
│   └── final_model.pth            # trained model + latent codes
├── distance_mat/
│   └── D_l2_updated.npy           # pairwise SDF distance matrix
└── README.md
```

Some paths in the notebooks may need to be adjusted depending on where the data and checkpoint files are stored.

---

## Phase 1: Training the shape model

`Phase_1_code_Thesis.ipynb` performs the full preprocessing and training pipeline.

Main steps:

1. **Load and pad binary wing masks**

   Images are loaded as grayscale, binarized, and padded to a square canvas of size `S = 1001`.

2. **Move images to a normalized coordinate grid**

   The pixel grid is converted to coordinates in `[-1,1] x [-1,1]`.

3. **Compute geometric normalization**

   For each wing, the code computes:

   - centroid,
   - covariance matrix,
   - size from the covariance trace,
   - principal-axis rotation,
   - scale,
   - translation.

   This gives a canonical aligned representative of each wing.

4. **Compute signed distance functions**

   Each aligned binary mask is converted into an SDF using `scipy.ndimage.distance_transform_edt`.

5. **Compute pairwise SDF distances**

   The notebook constructs an \(L^2\)-distance matrix between normalized SDFs.

6. **Train a SIREN-based DeepSDF model**

   The model takes as input:

   - a learned latent code \(z_i\) for a wing,
   - a coordinate \((x,y)\),

   and predicts the SDF value at that coordinate.

7. **Save outputs**

   The notebook saves:

   ```text
   model_files/final_model.pth
   distance_mat/D_l2_updated.npy
   ```

---

## Phase 2: Shape matching

`Phase_2_code_Thesis.ipynb` uses the trained model to solve a matching problem.

Given a new target image, the code searches for the best wing shape and pose by optimizing:

- latent code `z`,
- scale `lambda`,
- rotation angle,
- translation vector `t`.

The matching loss compares smooth characteristic functions derived from the target SDF and the transformed predicted SDF.  
The code also uses near-contour weighting and latent-space regularization to encourage realistic reconstructions.

The final result includes:

- the optimized latent code,
- the best similarity transform,
- an overlay plot of the target wing and reconstructed wing,
- a PCA plot showing where the optimized latent code lies relative to the training latent space.

---

## Main dependencies

The notebooks use:

```text
numpy
scipy
scikit-image
scikit-learn
matplotlib
Pillow
torch
torchvision
```

A typical installation command is:

```bash
pip install numpy scipy scikit-image scikit-learn matplotlib pillow torch torchvision
```

If using CUDA, install the PyTorch version that matches your CUDA setup from the official PyTorch instructions.

---

## How to run

### 1. Prepare the data

Place the binary wing masks in:

```text
test1/mask/
```

The masks should be binary or grayscale images where the wing is the foreground.

### 2. Run Phase 1

Open and run:

```text
Phase_1_code_Thesis.ipynb
```

This will:

- align the training wings,
- compute their SDFs,
- train the SIREN/DeepSDF model,
- save the trained model and latent codes.

### 3. Run Phase 2

Open and run:

```text
Phase_2_code_Thesis.ipynb
```

Make sure the checkpoint path is correct.  
The notebook currently expects a trained model checkpoint, for example:

```python
checkpoint_path = "final_model.pth"
```

or, depending on your folder structure:

```python
checkpoint_path = "model_files/final_model.pth"
```

Then choose the target image to reconstruct:

```python
image = "rec_16.png"
```

---

## Important implementation notes

### Coordinate convention

The code uses a square coordinate grid:

```text
[-1,1] x [-1,1]
```

The image is flipped vertically when moved to the coordinate grid so that the mathematical \(y\)-axis points upward.

### Similarity transformations

The code represents a similarity transform using:

```python
lambda, R, t
```

where:

- `lambda` is scale,
- `R` is a 2D rotation matrix,
- `t` is translation.

When transforming SDF predictions, the model is evaluated by inverse-mapping coordinates back to the learned normalized space.

### SDF convention

The SDF is computed as:

```text
outside distance - inside distance
```

Therefore:

- negative values are inside the wing,
- positive values are outside the wing,
- zero is the wing boundary.

### Latent space

Each training wing has a learned latent code.  
The model is regularized so that distances between latent codes reflect pairwise SDF distances between wings.

---

## Outputs

Typical outputs include:

- aligned wing visualizations,
- SDF contour plots,
- pairwise distance matrices,
- trained model checkpoint,
- PCA plots of the learned latent space,
- final shape-matching overlays.

---

## Notes

This code was written for thesis experiments and is organized as research code rather than as a polished Python package.  
For reuse, the most important things to check are:

1. file paths,
2. image resolution `S`,
3. checkpoint location,
4. target image path,
5. whether `my_siren_model.py` is included and matches the saved checkpoint.

---

## Author

Yoav Kamir  
Master's thesis project, University of Copenhagen
