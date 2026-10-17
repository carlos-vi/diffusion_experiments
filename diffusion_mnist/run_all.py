"""Runs all five steps in order. Usage: python run_all.py [--quick]

--quick trains for only 300 steps (about a minute) to check that everything works.
The digits will look bad; the full run (4000 steps) gives recognisable ones.
"""

import subprocess
import sys

quick = "--quick" in sys.argv
steps = [("step1_forward_process", []),
         ("step2_train", ["--steps", "300"] if quick else []),
         ("step3_sample", []),
         ("step4_fast_sampling", []),
         ("step5_guidance", [])]
for script, extra in steps:
    print(f"\n=== {script} ===", flush=True)
    subprocess.run([sys.executable, f"{script}.py", *extra], check=True)
print("\nDone. Open the 'results' folder.")
