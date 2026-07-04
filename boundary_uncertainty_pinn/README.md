# Boundary-uncertainty PINN experiment project

This directory is an isolated second-paper experiment workspace. It must not
modify `/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master`.

The experiment tests one mechanism:

- Generate displacement observations from a layered heterogeneous stiffness
  field with a true boundary/load amplitude.
- Train a PINN-like inverse model that either fixes a wrong amplitude or learns
  the amplitude jointly with the material field.
- Check whether boundary/load amplitude error is absorbed into the recovered
  stiffness field when it is not explicitly represented.
- Add minimal auxiliary scale information, especially reaction constraints and
  sparse boundary displacement anchors, to test whether identifiability can be
  restored.

Historical smoke tests are kept under `outputs/`. New experiments are written
under `exp/<experiment_group>/<condition>/<case>/<run_name>/`.

Suggested groups:

- `00.Smoke`: minimal remote-only checks before any long run.
- `01.BoundaryIdentifiability`: main paper comparison, oracle / wrong fixed /
  learnable boundary scale / reaction plus anchors.
- `02.AnchorAblation`: number and placement of boundary anchors.
- `03.ReactionAblation`: reaction scale constraints.
- `04.NoiseRobustness`: noisy displacement observations.
- `05.ObservationSparsity`: sparse interior displacement observations.

Each run stores `txt`, `png`, `metrics`, `json`, `model`, `npz`, and `dat`
subdirectories. The `png` directory includes loss curves, amplitude convergence,
material/displacement/stress fields, error maps, sampling layout, boundary
curves, reaction curves, material slices, and deformed-grid comparison.
