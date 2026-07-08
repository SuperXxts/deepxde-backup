# PowerShell 调远程集群固定规则

本项目本地终端是 Windows PowerShell，远端管理节点和计算节点是 Linux shell。以后所有远程实验启动、检查、同步和临时脚本都按本文执行，避免 PowerShell 把本应交给远端 bash 或 Python 的符号提前解析。

## 为什么会被 PowerShell 卡住

PowerShell 会先解析本地命令行，再把字符串传给 `ssh`。下面这些符号如果直接写进 `ssh manage "..."`，很容易先被本地 PowerShell 处理：

```text
|、||、&&
$var
<<EOF / <<'PY'
单双引号混用
awk、grep、sed、python -c 里带花括号或引号的表达式
```

典型后果包括：

```text
The token '||' is not a valid statement separator
The '<' operator is reserved for future use
远端 bash: 语法错误: 未预期的文件结尾
远端 Python: SyntaxError / EOF while scanning triple-quoted string
```

## 禁止写法

不要把复杂 bash 逻辑塞进 PowerShell 双引号：

```powershell
ssh manage "cd /path && for r in B0 B1; do ps -ef | grep $r || true; done"
```

不要在 PowerShell 里直接写 Linux here-doc：

```powershell
ssh manage "python3 - <<'PY'
print('hello')
PY"
```

不要用本地 `sed -i "s/\r$//"` 处理远端换行。以前这个写法把路径里的 `deepxde-master` 误伤成了 `deepxde-maste`。换行统一用远端 Python 处理。

## 推荐写法一：优先使用远端项目脚本

能固化的启动、检查、汇总逻辑，必须放进远端项目的 `hpc/*.sh`，本地只调用一个简单命令。

```powershell
ssh manage 'cd /public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn && bash hpc/check_fem_multiload_gate_lf.sh 02.FEMThreeLoadFormal 20260708_120000'
```

新增或修改脚本后必须检查：

```powershell
ssh manage 'cd /public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn && bash -n hpc/run_fem_multiload_gate_onecase_lf.sh hpc/launch_fem_multiload_gate_formal_lf.sh'
```

## 推荐写法二：简单远程命令用单引号

如果只是 `ls`、`tail`、`find` 这类简单命令，用 PowerShell 单引号包住远端命令，避免本地展开 `$变量`。

```powershell
ssh manage 'cd /public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn && find exp -maxdepth 2 -type d | sort | head'
```

包含远端变量时也用单引号：

```powershell
ssh manage 'for r in B0 B1 B2; do echo "$r"; done'
```

## 推荐写法三：复杂临时逻辑走 stdin

临时 Python 用 PowerShell here-string 通过 stdin 传给远端 Python：

```powershell
@'
from pathlib import Path
base = Path("/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn")
for rel in ["scripts/train_fem_multiload_gate.py", "src/bupinn/decoupling.py"]:
    p = base / rel
    print(rel, p.exists(), p.stat().st_size if p.exists() else None)
'@ | ssh manage python3 -
```

临时 bash 也通过 stdin：

```powershell
@'
set -euo pipefail
cd /public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
python -m py_compile scripts/train_fem_multiload_gate.py src/bupinn/decoupling.py
bash -n hpc/run_fem_multiload_gate_onecase_lf.sh
'@ | ssh manage bash -s
```

如果 stdin 脚本里还要二次 `ssh comput*`，不要再嵌套复杂引号。优先调用远端已有 `hpc/*.sh`。

不要在本地或管理节点 Python 里用下面这种方式传多行脚本：

```python
subprocess.run(["ssh", "comput1", "bash", "-lc", multi_line_script])
```

这类命令容易让远端 shell 把 `multi_line_script` 的第一段当成 `bash -lc` 的命令字符串，其余行又被外层 shell 继续解释，表现为无故打印整个 `set` 环境或出现半执行状态。需要临时向计算节点发送多行脚本时，使用 `ssh comput1 bash -s` 并通过 stdin 传入，或者直接新增/调用项目 `hpc/*.sh`。

## 远端换行规范

同步本地文件到远端后，如需保证 LF，使用远端 Python：

```powershell
@'
from pathlib import Path
base = Path("/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn")
for rel in [
    "hpc/run_fem_multiload_gate_onecase_lf.sh",
    "hpc/launch_fem_multiload_gate_formal_lf.sh",
]:
    p = base / rel
    text = p.read_text(encoding="utf-8", errors="replace")
    p.write_text(text.replace("\r\n", "\n").replace("\r", "\n"), encoding="utf-8")
'@ | ssh manage python3 -
```

不要用 `sed -i` 做 CRLF 修复。

## 实验状态检查规则

判断长实验是否仍在运行，不能只看 `screen`。必须至少同时检查：

```text
1. 计算节点上存在活的 Python 训练进程。
2. train.log 修改时间在增长。
3. 日志中的 step 或 DeepXDE 迭代数在增长。
```

不要把旧 `screen.log`、失败快照目录或历史日志当成 live 状态。

## 启动正式实验前的固定顺序

```text
1. 同步代码到远端项目目录。
2. 用远端 Python 统一 LF 换行。
3. 远端执行 py_compile 和 bash -n。
4. 先跑 smoke，确认代码、环境、产物目录和图表生成正常。
5. smoke 通过后再启动正式长跑。
6. 长跑启动后给出预计完成时间，不持续空转监控。
```

## 当前项目关键默认设置

正式 FEM 多工况主线默认不启用材料平滑项：

```text
MATERIAL_SMOOTHNESS_WEIGHT=0.0
```

动态损失图和材料参数演化图仍按 1000 步保存：

```text
DISPLAY_EVERY=1000
DYNAMIC_FIGURE_EVERY=1000
MATERIAL_MONITOR_EVERY=1000
```
