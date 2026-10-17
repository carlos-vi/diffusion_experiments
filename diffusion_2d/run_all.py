"""Runs all five steps in order. Usage: python run_all.py [moons|swiss]"""

import subprocess
import sys

dataset = sys.argv[1] if len(sys.argv) > 1 else "moons"
for script in ["step1_forward_process", "step2_train", "step3_sample",
               "step4_ddim_vs_ddpm", "step5_vector_field"]:
    print(f"\n=== {script} ({dataset}) ===", flush=True)
    subprocess.run([sys.executable, f"{script}.py", "--dataset", dataset], check=True)
print("\nDone. Open the 'results' folder.")
