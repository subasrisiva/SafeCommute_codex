import subprocess, sys, os
BASE = os.path.dirname(os.path.abspath(__file__))
steps = ["phase1_2_dataset.py", "phase3_5_eda_features.py", "phase6_7_model.py", "tn_network.py"]
for s in steps:
    print(f"\n=== Running {s} ===")
    if subprocess.run([sys.executable, os.path.join(BASE, s)], cwd=BASE).returncode != 0:
        print(f"FAILED at {s}"); sys.exit(1)
print("\nAll done. Open tn_safecommute_app.html with Live Server.")
