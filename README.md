# Tiny diffusion model, step by step

A diffusion model small enough to run on any laptop, built to understand how the idea works.
Instead of images it generates **2D points** (two moons, or a swiss roll), so every stage of
the process can be plotted and looked at. The network has ~38k parameters and the whole
pipeline runs in under a minute on CPU.

## Run it

```bash
git clone https://github.com/carlos-vi/bio_research_experiments.git
cd bio_research_experiments
python -m venv .venv && source .venv/bin/activate     # optional, on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py            # two moons; use `python run_all.py swiss` for the swiss roll
```

Everything is written to `results/`. The repository already contains the results of a run,
so you can look at them before running anything.

**Low RAM?** The default `torch` wheel can be large. The CPU-only build is lighter:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib pillow
```

Measured peak memory with the default (CUDA) wheel was about 700-850 MB per step, and roughly
500 MB of that is just `import torch`. The model and data themselves use a few MB.

## The idea in five lines

1. **Forward process:** repeatedly add a little Gaussian noise to a data point until only noise is left.
   This needs no learning, and there is a closed form to jump straight to any step `t`.
2. **Training:** take a clean point, noise it to a random step `t`, and train a network to
   predict *which noise* was added (a simple MSE loss).
3. **Reverse process:** start from pure noise and apply the network `T` times, each time
   removing a bit of the predicted noise. Out comes a new data point.
4. **Faster sampling:** DDIM reuses the same trained network but takes far fewer steps.
5. **What was learned:** predicted noise points away from the data, so minus the predicted
   noise is an arrow towards the data (this is the "score").

## Steps and what to look at

| Script | Output in `results/` | What to look for |
|---|---|---|
| `step1_forward_process.py` | `1_forward_process_*.png`, `1_noise_schedule.png` | The shape fades into a Gaussian blob. `alpha_bar` goes from 1 to about 0. |
| `step2_train.py` | `2_training_loss_*.png`, `model_*.pt` | Loss drops quickly, then plateaus. It never reaches 0, because the noise is random and cannot be predicted exactly. |
| `step3_sample.py` | `3_real_vs_generated_*.png`, `3_reverse_process_*.gif` | The GIF is the reverse process: noise condenses into the moons. |
| `step4_ddim_vs_ddpm.py` | `4_ddim_vs_ddpm_*.png` | Quality as a function of the number of steps (see below). |
| `step5_vector_field.py` | `5_vector_field_*.png` | At high `t` the arrows point to the overall blob; at low `t` they snap to the thin curves. |

Each script can also be run alone (`python step2_train.py --dataset swiss`). Steps 3 to 5 need
the model from step 2.

## A reference run (two moons)

Error is the mean distance from each generated point to its nearest real point (lower is better):

| Sampler | Network calls | Error |
|---|---|---|
| DDIM | 2 | 2.80 |
| DDIM | 5 | 0.13 |
| DDIM | 20 | 0.04 |
| DDIM | 100 | 0.034 |
| DDPM | 1000 | 0.032 |

20 DDIM steps get close to the 1000-step DDPM result. With 2 steps the result is garbage:
the first jump leaves the points far from anything the network saw during training, and
there is no second chance to correct it. Exact numbers vary slightly with the PyTorch version.

## Things to try

- Change `beta_end` or `T` in `tinydiff/schedule.py` and see how step 1 and the samples change.
- Shrink the network (`hidden` in `tinydiff/model.py`) or train for fewer steps (`--steps 500`)
  and watch the samples degrade.
- Add your own dataset to `tinydiff/data.py`.

## Layout

```
tinydiff/
  data.py       toy datasets
  schedule.py   noise schedule and forward process
  model.py      time-conditioned MLP that predicts the noise
  sampling.py   DDPM and DDIM samplers
  metrics.py    nearest-neighbour error
step1..step5    one script per stage
run_all.py      runs them all
```
