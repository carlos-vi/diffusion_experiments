"""Runs all five steps in order. Usage: python run_all.py [--quick] [options for step2_train.py]

--quick trains for only 300 steps (about a minute) to check that everything works.
The digits will look bad; the full run (5000 steps) gives recognisable ones.
Any other option is passed on to step2_train.py, e.g. on a GPU:

    python run_all.py --steps 20000 --base 64 --attn --batch 256 --amp
"""

import subprocess
import sys

train_args = ["--steps", "300"] if "--quick" in sys.argv else []
train_args += [a for a in sys.argv[1:] if a != "--quick"]
steps = [("step1_forward_process", []),
         ("step2_train", train_args),
         ("step3_sample", []),
         ("step4_fast_sampling", []),
         ("step5_guidance", [])]
for script, extra in steps:
    print(f"\n=== {script} ===", flush=True)
    subprocess.run([sys.executable, f"{script}.py", *extra], check=True)
print("\nDone. Open the 'results' folder.")
