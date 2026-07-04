"""Formal ablation case definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CaseDefinition:
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
            "net": self.net,
            "top": self.top,
            "learnable_A": self.learnable_A,
            "reaction": self.reaction,
            "anchor": self.anchor,
            "boundary_layer": self.boundary_layer,
            "decoupled_obs": self.decoupled_obs,
            "wide": self.wide,
            "case_id": self.case_id,
            "short_name": self.short_name,
            "purpose": self.purpose,
        }


CASE_DEFINITIONS: tuple[CaseDefinition, ...] = (
    CaseDefinition(
        "A0",
        "PFNN-正确边界",
        net="pfnn",
        top="true",
        purpose=(
            "标准 PFNN，顶部竖向位移边界采用真实幅值 A=1.0，底部固定 u=0、v=0，"
            "顶部水平位移 u=0，左右边界不显式给定，使用普通位移观测损失。"
            "作用是给出边界完全正确时的理想基线。"
        ),
    ),
    CaseDefinition(
        "A1",
        "PFNN-错误固定边界",
        net="pfnn",
        top="wrong",
        purpose=(
            "标准 PFNN，顶部竖向位移边界被错误固定为 A=0.85，其余边界同 A0，"
            "使用普通位移观测损失。作用是验证错误边界是否会污染 K、mu 材料场反演。"
        ),
    ),
    CaseDefinition(
        "A2",
        "PFNN-可学习边界幅值",
        net="pfnn",
        top="learnable_A",
        learnable_A=True,
        purpose=(
            "标准 PFNN，顶部竖向位移写成 A 乘以已知形状函数，A 从 0.85 初始化并参与训练，"
            "不加入反力和锚点。作用是验证仅学习一个边界幅值是否足以修正错误边界。"
        ),
    ),
    CaseDefinition(
        "A3",
        "PFNN-可学习边界幅值-反力",
        net="pfnn",
        top="learnable_A",
        learnable_A=True,
        reaction=True,
        purpose=(
            "标准 PFNN，顶部边界幅值 A 可学习，并加入顶部总反力积分约束；无边界锚点。"
            "作用是验证全局反力信息能否改善刚度尺度识别。"
        ),
    ),
    CaseDefinition(
        "A4",
        "PFNN-可学习边界幅值-锚点",
        net="pfnn",
        top="learnable_A",
        learnable_A=True,
        anchor=True,
        purpose=(
            "标准 PFNN，顶部边界幅值 A 可学习，并加入少量边界位移锚点；无总反力约束。"
            "作用是单独评估边界位移锚点对边界识别和材料反演的贡献。"
        ),
    ),
    CaseDefinition(
        "A5",
        "PFNN-可学习边界幅值-锚点-反力",
        net="pfnn",
        top="learnable_A",
        learnable_A=True,
        reaction=True,
        anchor=True,
        purpose=(
            "标准 PFNN，顶部边界幅值 A 可学习，同时加入少量边界位移锚点和顶部总反力积分约束。"
            "作用是给出没有新网络结构和解耦训练器时的最强 PFNN 辅助信息基线。"
        ),
    ),
    CaseDefinition(
        "A6",
        "宽PFNN-可学习边界幅值-反力",
        net="pfnn",
        top="learnable_A",
        learnable_A=True,
        reaction=True,
        wide=True,
        purpose=(
            "加宽加深的 PFNN，顶部边界幅值 A 可学习，并加入总反力约束。"
            "作用是参数量控制，排除最终方法提升只是因为网络更大的可能。"
        ),
    ),
    CaseDefinition(
        "A7",
        "分支网络-错误固定边界",
        net="split",
        top="wrong",
        purpose=(
            "采用状态分支和材料分支，但顶部边界仍错误固定为 A=0.85，不使用边界分支、反力、锚点和解耦。"
            "作用是验证仅拆分状态场和材料场是否足以解决边界错误污染问题。"
        ),
    ),
    CaseDefinition(
        "A8",
        "分支网络-可学习边界幅值",
        net="split",
        top="learnable_A",
        learnable_A=True,
        purpose=(
            "采用状态分支和材料分支，顶部边界幅值 A 可学习，但不使用边界分支和解耦。"
            "作用是区分网络分支结构与边界幅值可学习这两个因素。"
        ),
    ),
    CaseDefinition(
        "A9",
        "分支网络-可学习边界幅值-反力",
        net="split",
        top="learnable_A",
        learnable_A=True,
        reaction=True,
        purpose=(
            "采用状态分支和材料分支，顶部边界幅值 A 可学习，并加入总反力约束。"
            "作用是评估状态/材料分支在传统可学习边界框架中的收益。"
        ),
    ),
    CaseDefinition(
        "A10",
        "边界分支-普通观测损失",
        net="split",
        top="network_top",
        boundary_layer=True,
        purpose=(
            "采用状态分支、材料分支和低维边界分支，顶部竖向位移由边界分支学习；"
            "仍使用普通位移观测损失，不加入反力、锚点和解耦。"
            "作用是单独评估边界分支本身的贡献。"
        ),
    ),
    CaseDefinition(
        "A11",
        "边界分支-普通观测损失-反力",
        net="split",
        top="network_top",
        boundary_layer=True,
        reaction=True,
        purpose=(
            "在 A10 基础上加入顶部总反力积分约束，但仍不使用解耦训练器和边界锚点。"
            "作用是评估反力信息与边界分支的组合效果。"
        ),
    ),
    CaseDefinition(
        "A12",
        "边界分支-普通观测损失-锚点",
        net="split",
        top="network_top",
        boundary_layer=True,
        anchor=True,
        purpose=(
            "在 A10 基础上加入少量边界位移锚点，但仍不使用解耦训练器和总反力。"
            "作用是单独评估锚点信息与边界分支的组合效果。"
        ),
    ),
    CaseDefinition(
        "A13",
        "边界分支-解耦训练器",
        net="split",
        top="network_top",
        boundary_layer=True,
        decoupled_obs=True,
        purpose=(
            "采用边界分支和解耦训练器，不加入反力和锚点。"
            "作用是单独验证残差投影解耦是否能减少材料场替边界误差背锅。"
            "正式诊断中需要区分旧 mixed 平衡形式和新 direct 材料平衡形式。"
        ),
    ),
    CaseDefinition(
        "A14",
        "边界分支-解耦训练器-反力",
        net="split",
        top="network_top",
        boundary_layer=True,
        decoupled_obs=True,
        reaction=True,
        purpose=(
            "采用边界分支和解耦训练器，并加入顶部总反力积分约束；无边界锚点。"
            "作用是验证解耦后全局反力能否进一步恢复材料刚度尺度。"
            "正式方法中反力应优先使用由位移梯度和 K、mu 计算的本构应力积分，而不是仅使用网络应力分支。"
        ),
    ),
    CaseDefinition(
        "A15",
        "最终方法-边界分支-解耦-锚点-反力",
        net="split",
        top="network_top",
        boundary_layer=True,
        decoupled_obs=True,
        reaction=True,
        anchor=True,
        purpose=(
            "最终方法：状态分支、材料分支、低维边界分支、解耦训练器、少量边界位移锚点和顶部总反力积分约束。"
            "作用是验证完整框架能否同时恢复边界幅值和 K、mu 空间材料场。"
            "旧 mixed 平衡 + 应力分支反力只作为失败消融，新 direct 平衡 + 本构反力作为候选正式方法。"
        ),
    ),
)

CASE_CHOICES = tuple(case.case_id for case in CASE_DEFINITIONS)
CASE_TABLE = {case.case_id: case for case in CASE_DEFINITIONS}


def case_settings(case_id: str) -> dict[str, object]:
    return CASE_TABLE[case_id].settings()


def case_description(case_id: str) -> str:
    case = CASE_TABLE[case_id]
    return f"{case.case_id} {case.short_name}\n\n{case.purpose}\n"


def case_markdown_table() -> str:
    lines = [
        "| 组别 | 简称 | 网络 | 顶部竖向边界 | 解耦 | 反力 | 锚点 | 目的 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    top_names = {
        "true": "真实 A=1.0",
        "wrong": "错误固定 A=0.85",
        "learnable_A": "标量 A 可学习",
        "network_top": "边界分支学习",
    }
    for case in CASE_DEFINITIONS:
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
