# Project rules

1. Do not modify `/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master`.
2. Run PINN training only on the remote GPU/HPC nodes, never on the local desktop.
3. Run smoke checks before formal long experiments.
4. Live status requires process, log timestamp, and step growth checks; screen existence alone is insufficient.
5. Preserve failed run directories before relaunching same-name experiments.
6. Store every run under `exp/<group>/<run_name>/` with `txt`, `png`, `metrics`, `json`, `model`, `npz`, and `dat`.
7. Keep experiments reproducible: save config, seed, node, device, code version notes, loss history, metrics, model weights, and plots.
8. When launching or checking remote HPC jobs from a local PowerShell shell, do not send complex inline ssh commands containing `|`, `||`, `$var`, here-docs, or nested quotes. Use the safe patterns in `docs/powershell_ssh_runbook.md`: project shell scripts, single-quoted simple remote commands, or script-over-stdin.
