# CLA Project

This repository contains a computational linear algebra project for streaming background subtraction using an incremental thin SVD approach. The goal is to model a video sequence as a stream of frame vectors and maintain a low-rank approximation of the background while separating moving foreground objects.

The project combines numerical linear algebra with a practical computer vision application. It implements incremental updates to a low-rank factorization, stability checks, and repair mechanisms to keep the decomposition numerically reliable as new frames arrive.

## Project Overview

The codebase treats each video frame as a column vector and maintains a rank-truncated representation of the stream. The main idea is:

- Each frame is flattened into a vector.
- A low-rank factorization is updated incrementally as new frames arrive.
- The low-rank component is treated as the background.
- The residual between the current frame and the background is treated as foreground motion.

This is implemented in a numerical linear algebra setting with custom routines for:

- incremental SVD updates,
- rank growth and truncation,
- orthogonality drift tracking,
- repair of numerically unstable updates,
- adaptive trigger-based correction.

## Repository Structure

- `isvd.py` — incremental thin SVD implementation and streaming matrix update logic.
- `stab.py` — stability metrics, repair routines, and adaptive trigger logic.
- `vid.py` — OpenCV-based video pipeline for converting videos into low-rank background/foreground decomposition.
- `exp.py` — experimental drift/stability analysis for the tracking algorithm.
- `grid_search.py` — grid search / hyperparameter exploration script.
- `ablation.py` — ablation study for different algorithm configurations.
- `test_isvd.py` — unit tests verifying correctness and stability.
- `results/` — generated result artifacts and experiment outputs.
- `report.pdf` — project report.
- `*.mp4` files — demo/output videos.
- `drift.png` — visual summary of drift behavior.

## Key Algorithms

### Incremental Thin SVD
`isvd.py` contains the core streaming SVD implementation. It maintains a decomposition of the form:

- X = U @ Up @ diag(sv) @ Vp.T @ V.T

where:

- `U` and `V` grow over time,
- `Up`, `sv`, and `Vp` absorb rotations and updates,
- the method keeps the rank representation compact and efficient for streaming use.

### Stability and Repair
`stab.py` adds a second layer of numerical robustness:

- orthogonality drift tracking,
- adaptive repair checks,
- modified Gram-Schmidt (MGS) based re-orthogonalization,
- trigger-based correction when the factorization drifts beyond a tolerance.

### Background Subtraction Pipeline
`vid.py` shows how to apply the tracker to a video stream:

- read frames,
- resize and grayscale them,
- flatten each frame into a vector,
- update the low-rank tracker,
- reconstruct the background,
- compute the foreground residual.

## Requirements

This project uses Python with the following dependencies:

- Python 3.9+
- NumPy
- OpenCV (`opencv-python`)
- Matplotlib

Install dependencies with:

```bash
pip install numpy matplotlib opencv-python
```

## Usage

### Create a demo video

```bash
python vid.py demo.avi --make-demo
```

### Run the background subtraction pipeline

```bash
python vid.py demo.avi --out result.avi
```

This writes a video with three panels:

1. original frame,
2. reconstructed background,
3. foreground residual.

### Run experiments

```bash
python exp.py
```

This generates drift and stability plots, and reports reconstruction error.

### Run tests

```bash
pytest -q
```

or:

```bash
python test_isvd.py
```

## Example Outputs

The repository includes generated artifacts such as:

- `drift.png`
- `vault_demo.mp4`
- `robbery_input.mp4`
- `robbery_output.mp4`

These illustrate the algorithm's behavior on synthetic and real-style scenes.

## Notes

This project is primarily a computational linear algebra and numerical optimization exercise, with a practical computer vision application. The implementation is intentionally lightweight and focused on understanding how streaming low-rank updates can be stabilized in the presence of drift and numerical error.

## Related Materials

- `report.pdf` contains the project write-up and analysis.
- `results/` stores additional experiment outputs and artifacts.

## Summary

This repository demonstrates how a streaming low-rank tracker can be used to maintain a stable, low-rank approximation of a changing data stream while retaining enough fidelity to detect moving objects. The overall approach combines incremental SVD updates, stability monitoring, and background subtraction into a single computational linear algebra workflow for video analysis.
