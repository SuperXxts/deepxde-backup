# Linear Elasticity Benchmark

This benchmark turns the current PINN elasticity experiments into a reusable comparison suite.

## Goals
- Fix the geometry, PDE, boundary conditions, and evaluation protocol.
- Make new methods comparable under the same training budget.
- Save metrics and best test loss in a consistent location.

## Current benchmark tasks
- `linear_forward_const`: 2D linear elasticity forward solution with constant material parameters.
- `linear_inverse_const`: infer constant `lambda` and `mu` from displacement observations.
- `linear_inverse_observe_sweep`: study inverse performance under different observation counts.

## Fixed benchmark protocol
- Geometry: unit square plate.
- Outputs: `ux`, `uy`, `Sxx`, `Syy`, `Sxy`.
- Primary metrics: relative L2 error for all fields.
- Secondary metrics: mean/max absolute error, best test loss.
- Reproducibility: save task config, command, run folder, and aggregated summary.

## Run examples
```bash
cd /public/home/xinxi/wxtian/WXTIAN/PINN/deepxde
python benchmark/run_benchmark.py --task benchmark/tasks/linear_forward_const.json
python benchmark/run_benchmark.py --task benchmark/tasks/linear_inverse_const.json
python benchmark/run_benchmark.py --task benchmark/tasks/linear_inverse_observe_sweep.json
python benchmark/summarize_benchmark.py
```

## Output layout
```text
benchmark/
  tasks/
  runs/
    <task_name>/
      <run_name>/
        metrics/accuracy_metrics.json
        json/best_test_loss.json
  summary/
    benchmark_summary.csv
    benchmark_summary.json
```

## How to extend
When a new method is added, keep the same:
- geometry
- train/test split size
- optimization budget
- metric definitions
- save schema

That way, PINN, PI-DeepONet, FEM+NN hybrids, and future variants can be compared fairly.
