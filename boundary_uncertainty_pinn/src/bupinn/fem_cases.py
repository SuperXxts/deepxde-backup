"""B-group FEM benchmark case definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FEMCaseDefinition:
    case_id: str
    short_name: str
    net: str
    top: str
    learnable_A: bool = False
    reaction: bool = False
    anchor: bool = False
    boundary_layer: bool = False
    decoupled_obs: bool = False
    wide: bool = False
    purpose: str = ""

    def settings(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "short_name": self.short_name,
            "net": self.net,
            "top": self.top,
            "learnable_A": self.learnable_A,
            "reaction": self.reaction,
            "anchor": self.anchor,
            "boundary_layer": self.boundary_layer,
            "decoupled_obs": self.decoupled_obs,
            "wide": self.wide,
            "purpose": self.purpose,
        }


FEM_CASE_DEFINITIONS: tuple[FEMCaseDefinition, ...] = (
    FEMCaseDefinition(
        "B0",
        "PFNN-正确边界-FEM",
        net="pfnn",
        top="true",
        purpose=(
            "FEM 地基加载板算例中的标准 DeepXDE PFNN，加载板竖向位移采用真实幅值。"
            "作用是给出工程化数值数据上的理想边界基线。"
        ),
    ),
    FEMCaseDefinition(
        "B1",
        "PFNN-错误固定边界-FEM",
        net="pfnn",
        top="wrong",
        purpose=(
            "标准 PFNN，训练时把加载板竖向位移幅值错误固定为真实值的 0.85。"
            "作用是验证错误边界在 FEM 岩土场景中是否会污染 K 和 mu 材料场反演。"
        ),
    ),
    FEMCaseDefinition(
        "B2",
        "PFNN-可学习边界幅值-FEM",
        net="pfnn",
        top="learnable_A",
        learnable_A=True,
        purpose=(
            "标准 PFNN，加载板竖向位移幅值 A 从 0.85 初始化并参与训练，不使用反力和锚点。"
            "作用是检验单独学习边界幅值能否恢复 FEM 材料反演。"
        ),
    ),
    FEMCaseDefinition(
        "B3",
        "PFNN-可学习边界幅值-反力-FEM",
        net="pfnn",
        top="learnable_A",
        learnable_A=True,
        reaction=True,
        purpose=(
            "标准 PFNN，A 可学习，并加入加载板总反力约束。"
            "作用是检验全局反力信息能否补充整体刚度尺度。"
        ),
    ),
    FEMCaseDefinition(
        "B4",
        "分支网络-可学习边界幅值-反力-FEM",
        net="split",
        top="learnable_A",
        learnable_A=True,
        reaction=True,
        purpose=(
            "状态分支和材料分支网络，A 可学习，并加入总反力。"
            "作用是排除最终效果只是来自状态/材料分支结构和反力组合。"
        ),
    ),
    FEMCaseDefinition(
        "B5",
        "边界分支-普通观测-反力-FEM",
        net="split",
        top="network_top",
        boundary_layer=True,
        reaction=True,
        purpose=(
            "状态分支、材料分支和低维边界分支共同训练，但位移观测仍采用普通残差，"
            "并加入加载板总反力。作用是检查不做解耦时边界分支是否仍与材料场混淆。"
        ),
    ),
    FEMCaseDefinition(
        "B6",
        "边界分支-解耦观测-反力-FEM",
        net="split",
        top="network_top",
        boundary_layer=True,
        decoupled_obs=True,
        reaction=True,
        purpose=(
            "采用低维边界分支、解耦观测残差和加载板总反力，不加入边界锚点。"
            "作用是验证解耦训练器在 FEM 场景中的核心贡献。"
        ),
    ),
    FEMCaseDefinition(
        "B7",
        "完整方法-FEM",
        net="split",
        top="network_top",
        boundary_layer=True,
        decoupled_obs=True,
        reaction=True,
        anchor=True,
        purpose=(
            "完整方法：状态分支、材料分支、低维边界分支、解耦观测残差、少量边界位移锚点和加载板总反力。"
            "作用是验证本文方法能否在 FEM 岩土主算例中同时恢复边界和空间变 K/mu 材料场。"
        ),
    ),
)

FEM_CASE_CHOICES = tuple(case.case_id for case in FEM_CASE_DEFINITIONS)
FEM_CASE_TABLE = {case.case_id: case for case in FEM_CASE_DEFINITIONS}


def fem_case_settings(case_id: str) -> dict[str, object]:
    return FEM_CASE_TABLE[case_id].settings()


def fem_case_description(case_id: str) -> str:
    case = FEM_CASE_TABLE[case_id]
    return f"{case.case_id} {case.short_name}\n\n{case.purpose}\n"


def fem_case_markdown_table() -> str:
    top_names = {
        "true": "真实加载板位移",
        "wrong": "错误固定幅值 0.85",
        "learnable_A": "标量 A 可学习",
        "network_top": "边界分支学习",
    }
    lines = [
        "| 组别 | 简称 | 网络 | 加载板竖向边界 | 解耦 | 反力 | 锚点 | 目的 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for case in FEM_CASE_DEFINITIONS:
        lines.append(
            "| "
            + " | ".join(
                [
                    case.case_id,
                    case.short_name,
                    case.net,
                    top_names.get(case.top, case.top),
                    "是" if case.decoupled_obs else "否",
                    "是" if case.reaction else "否",
                    "是" if case.anchor else "否",
                    case.purpose,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"
