# diffusion_experiments

Small, readable diffusion-model experiments that run on a laptop CPU.

| Folder | What it is | Run time |
|---|---|---|
| [`diffusion_2d`](diffusion_2d) | Diffusion on 2D points (two moons, swiss roll). The whole process can be plotted. | under a minute |
| [`diffusion_mnist`](diffusion_mnist) | A small class-conditional diffusion model on MNIST digits, with fast sampling and classifier-free guidance. | about 15 minutes on CPU |

Each folder is self-contained, with its own README, `requirements.txt` and `run_all.py`.
Start with `diffusion_2d` if diffusion models are new to you.
