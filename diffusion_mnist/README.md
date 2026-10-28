# Tiny diffusion model on MNIST

A small class-conditional diffusion model that generates handwritten digits. It is the
image version of [`../diffusion_2d`](../diffusion_2d), so read that one first if diffusion
models are new to you. The network is a U-Net with about 100k parameters, and everything runs on CPU.

## Run it

```bash
git clone https://github.com/carlos-vi/diffusion_experiments.git
cd diffusion_experiments/diffusion_mnist
python -m venv .venv && source .venv/bin/activate     # optional, on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py --quick    # about a minute: checks that everything works (digits will look bad)
python run_all.py            # the full run: 5000 training steps
```

MNIST (about 11 MB) is downloaded to `data/` on the first run. Run the scripts from inside this folder.
Everything is written to `results/`, which already contains the output of a full run.

The training length is `--steps` in `step2_train.py` (default 5000, which is what the committed results used).

**Time and RAM.** On a 4-core CPU the 5000-step training took about 11 minutes, and the whole
pipeline about 14 minutes. A weaker machine will be slower; lower `--steps` if needed.
Peak memory per step was 800-1000 MB with the default (CUDA) PyTorch wheel, of which about 500 MB is just
`import torch`. The model and data themselves are small (the images are kept as 47 MB of uint8).

**Low RAM?** The CPU-only build of PyTorch is lighter:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib pillow
```

## Run it on a free GPU

The code uses the GPU automatically when PyTorch can see one, and the CPU otherwise. The images are
only 47 MB, so what limits a bigger model is GPU memory (VRAM), not RAM. Free options (limits change, check each site):

| Platform | GPU | Notes |
|---|---|---|
| [Kaggle Notebooks](https://www.kaggle.com/code) | P100 16 GB or 2x T4 | About 30 GPU hours per week, long sessions. Needs a verified phone number. |
| [Google Colab](https://colab.research.google.com) | T4 16 GB | Shorter sessions, GPU not guaranteed. Can save to Google Drive. |
| [Lightning AI Studios](https://lightning.ai) | T4 and others | Monthly free credits. |

Open [`colab_kaggle.ipynb`](colab_kaggle.ipynb) there (upload it, or open it from GitHub), turn on a GPU and run the cells.
Or, in any terminal with a GPU:

```bash
python run_all.py --steps 20000 --base 64 --attn --batch 256 --amp
```

| Option of `step2_train.py` | Effect |
|---|---|
| `--base N` | U-Net width, a multiple of 8. 16 (default) is 100k parameters, 32 is 370k, 64 is 1.4M, 96 is 3.3M (with `--attn`). |
| `--attn` | Adds self-attention in the 7x7 bottleneck. |
| `--batch N` | Batch size; a GPU handles 256 or 512 easily. |
| `--amp` | Mixed precision (fp16): faster and uses less VRAM. Ignored on CPU. |

The chosen architecture is saved in `results/model_config.json`, so steps 3 to 5 rebuild the same network
(a `model.pt` without that file is the default small one). The sampling noise is drawn on the CPU and then moved,
so the same seed gives the same starting noise on CPU and GPU.

## What is different from the 2D version

- The network is a small U-Net (convolutions, down- and up-sampling, skip connections) instead of an MLP.
- It is told which digit to draw. The label is randomly dropped during training (10% of the time),
  so the same network can also draw without a label. This is what makes step 5 possible.
- An EMA copy of the weights (a slowly moving average) is what gets saved and used for sampling.

## Steps and what to look at

| Script | Output in `results/` | What to look for |
|---|---|---|
| `step1_forward_process.py` | `1_forward_process.png`, `1_noise_schedule.png` | Real digits turning into static. By t = 400 the digit is barely visible. |
| `step2_train.py` | `2_training_loss.png`, `model.pt` | The loss drops quickly and then plateaus. |
| `step3_sample.py` | `3_generated_digits.png`, `3_reverse_process.png`, `3_reverse_process.gif` | 64 digits generated without any label, and the denoising animation. |
| `step4_fast_sampling.py` | `4_fast_sampling.png` | The same starting noise denoised in 3, 5, 10, 25, 50 and 1000 steps. |
| `step5_guidance.py` | `5_guidance.png` | Rows are the requested digits 0 to 9, for increasing guidance strength. |

Each script can also be run alone. Steps 3 to 5 need the model from step 2.

## Results of the committed run

**Fast sampling** (12 images, 4-core CPU). DDIM reuses the same network with far fewer steps:

| Sampler | Network calls | Time |
|---|---|---|
| DDIM | 3 | under 0.1 s (images are blurry) |
| DDIM | 5 to 10 | about 0.1 s (recognisable) |
| DDIM | 50 | 0.5 s |
| DDPM | 1000 | about 10 s |

**Classifier-free guidance.** The model combines its label-free and labelled predictions as
`eps = eps_uncond + w * (eps_cond - eps_uncond)`. A separate small classifier (98% test accuracy,
trained once in about 20 seconds) checks whether each generated digit is the one that was asked for
(60 images per setting, DDIM with 50 steps):

| Guidance w | Matches the request |
|---|---|
| 0 (label ignored) | 12% (chance is 10%) |
| 1 (follow the label) | 77% |
| 3 | 100% |
| 6 | 100% |

Higher guidance gives cleaner, more typical digits and less variety: look at how similar the
rows in the `w = 6` panel are to each other.

The model is small and trained briefly, so the free-form digits (step 3) are often wobbly or
ambiguous. That is expected, and a bigger network or longer training would improve them.

## Things to try

- Train with `--steps 1000` and `--steps 10000` and compare the digits.
- Change the width of the U-Net (`--base` in `step2_train.py`) and add `--attn`.
- Try guidance values between 0 and 3 in `step5_guidance.py`.

## Layout

```
tinymnist/
  data.py        MNIST loading
  schedule.py    noise schedule and forward process
  device.py      picks the GPU if there is one, else the CPU
  model.py       time- and label-conditioned U-Net (configurable width, optional attention)
  sampling.py    DDPM and DDIM samplers, with classifier-free guidance
  classifier.py  small classifier used only to evaluate step 5
step1..step5     one script per stage
run_all.py       runs them all (extra options go to step2_train.py)
colab_kaggle.ipynb  notebook to run everything on a free Kaggle / Colab GPU (extra options go to step2_train.py)
colab_kaggle.ipynb  notebook to run everything on a free Kaggle / Colab GPU
```
