# PowerShell 调远程集群命令避坑记录

本文档专门记录本项目在 Windows PowerShell 本地终端中通过 `ssh manage` 调远程集群时的固定规则。目的很简单：避免 PowerShell 把本来应该交给远端 Linux shell 的 `|`、`||`、`$变量`、here-doc 或引号提前解析，导致浪费时间。

## 1. 根本原因

本地 shell 是 PowerShell，远端 shell 是 Linux bash。下面这些符号如果直接写在 PowerShell 命令字符串里，很容易先被 PowerShell 解释，而不是传给远端：

```text
|
||
$
<<EOF
单双引号混用
带 f-string 或花括号的 Python one-liner
```

因此，涉及远端循环、管道、grep、awk、Python here-doc、bash here-doc 的命令，不能直接写成长的 `ssh manage "..."`。

## 2. 禁止写法

不要在 PowerShell 里直接写这种命令：

```powershell
ssh manage "for n in ...; do ps ... | grep xxx || true; done"
```

不要写这种 here-doc：

```powershell
ssh manage "python3 - <<'PY'
print('hello')
PY"
```

不要在双引号远端命令里直接写远端变量：

```powershell
ssh manage "for r in A5 A13; do echo $r; done"
```

这些写法可能被 PowerShell 提前解析，常见报错包括：

```text
The token '||' is not a valid statement separator
The '<' operator is reserved for future use
Missing expression after ','
远端 bash: 语法错误: 未预期的文件结尾
```

## 3. 推荐写法一：优先使用项目脚本

能写成脚本的检查、启动、汇总逻辑，全部放到远端项目的 `hpc/*.sh` 里，然后本地只调用脚本。

推荐：

```powershell
ssh manage 'cd /public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn && bash hpc/check_key_ablation_lf.sh 00.SmokeKeyAblation 20260703_key_smoke3000'
```

优点：

```text
1. 本地 PowerShell 只负责传一个简单命令。
2. 复杂循环、管道、grep、awk 都在远端 bash 脚本里执行。
3. 脚本可以 bash -n 检查，可以复用，可以记录到版本中。
```

## 4. 推荐写法二：简单远端命令用单引号

如果必须直接 `ssh manage`，远端命令尽量用 PowerShell 单引号包住，避免本地展开 `$变量`。

推荐：

```powershell
ssh manage 'cd /path/to/project && for r in A5 A13 A14; do if [ -e "exp/$r" ]; then echo exists:$r; else echo clear:$r; fi; done'
```

不要用 PowerShell 双引号包住包含 `$r` 的远端命令。

## 5. 推荐写法三：复杂临时逻辑用 stdin 脚本

如果临时检查逻辑太长，不要塞进一行 `ssh manage "..."`。可以把脚本内容通过 stdin 传给远端解释器。

PowerShell 示例：

```powershell
@'
from pathlib import Path
base = Path("/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn")
for run in ["A5", "A13", "A14", "A15"]:
    print(run, (base / "exp" / "00.SmokeKeyAblation" / run).exists())
'@ | ssh manage python3 -
```

或者：

```powershell
@'
set -euo pipefail
cd /public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn
bash hpc/check_key_ablation_lf.sh 00.SmokeKeyAblation 20260703_key_smoke3000
'@ | ssh manage bash -s
```

注意：stdin 脚本里如果再次嵌套 ssh 到计算节点，也要尽量把复杂逻辑放进远端已有脚本，避免多层引号。

## 6. 中文文件名检查规则

PowerShell 管道有时会让中文文件名在传入远端 Python 时变成问号。检查中文图名时，不直接把中文文件名嵌在本地命令里，优先使用 Unicode 转义或远端脚本。

推荐：

```python
name = "\u635f\u5931\u5386\u53f2\u56fe.png"  # 损失历史图.png
```

或者直接在远端项目中写检查脚本。

## 7. 长实验状态检查固定流程

判断实验是否 live，仍然必须同时看三类信息：

```text
1. 计算节点上是否存在活的 python 训练进程。
2. train.log 的修改时间是否是当前时间。
3. 日志中的 step 是否增长。
```

不能只看 screen，不能用旧日志，不能用历史 `screen.log`。

本项目优先使用：

```powershell
ssh manage 'cd /public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn && bash hpc/check_key_ablation_lf.sh <GROUP> <STAMP>'
```

## 8. 后续执行原则

以后涉及远端实验，默认顺序是：

```text
1. 能用 hpc 脚本就用 hpc 脚本。
2. 必须临时检查时，用 PowerShell 单引号包住简单远端命令。
3. 命令包含管道、循环、Python 多行、here-doc 时，用 stdin 脚本。
4. 不再在 PowerShell 双引号里硬塞复杂 bash/Python。
5. 新增的远端脚本必须先做 bash -n；Python 文件必须先 py_compile。
```
