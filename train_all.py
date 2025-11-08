"""
Train all 4 models for comparison:

1. Baseline (clean)

2. Baseline + Attacks (vulnerability demonstration)

3. DPRL (sensor noise robustness)

4. YOUR METHOD (complete adversarial robustness)

"""

import os
import subprocess
import time

def train_all_models():
    """Train all models sequentially"""
    
    experiments = [
        {
            'name': 'Baseline',
            'cmd': 'python train_complete.py --mode baseline --max-episodes 500 --save-dir ./experiments/baseline'
        },
        {
            'name': 'Baseline + Attacks',
            'cmd': 'python train_complete.py --mode baseline_attacked --max-episodes 500 --save-dir ./experiments/baseline_attacked'
        },
        {
            'name': 'DPRL (Privileged Learning)',
            'cmd': 'python train_complete.py --mode dprl --max-episodes 500 --add-sensor-noise --save-dir ./experiments/dprl'
        },
        {
            'name': 'Robust (YOUR METHOD)',
            'cmd': 'python train_complete.py --mode robust --max-episodes 1000 --enable-all-defenses --save-dir ./experiments/robust'
        }
    ]
    
    print("="*80)
    print("TRAINING ALL MODELS")
    print("="*80)
    print(f"Total experiments: {len(experiments)}")
    print(f"Estimated time: 8-12 hours")
    print("="*80 + "\n")
    
    for i, exp in enumerate(experiments):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i+1}/{len(experiments)}: {exp['name']}")
        print(f"{'='*80}")
        print(f"Command: {exp['cmd']}\n")
        
        start_time = time.time()
        
        # Run training
        result = subprocess.run(exp['cmd'], shell=True)
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n[OK] {exp['name']} completed successfully in {elapsed/3600:.2f} hours")
        else:
            print(f"\n[ERROR] {exp['name']} failed with return code {result.returncode}")
            break
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Compare results: python compare_results.py")
    print("  2. Generate paper figures: python generate_figures.py")

if __name__ == '__main__':
    train_all_models()

