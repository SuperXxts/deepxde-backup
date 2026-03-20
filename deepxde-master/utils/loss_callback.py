"""
实时保存loss可视化的Callback
用于在训练过程中定期更新和保存loss历史图像
"""
import os
import deepxde as dde
from .save_results import plot_and_save_loss_history, plot_all_loss_components, plot_parameter_history, save_loss_history_json, save_best_test_loss_json


class ParameterPlottingCallback(dde.callbacks.Callback):
    """
    在训练过程中实时绘制参数变化曲线的Callback
    该Callback会定期读取 VariableValue 生成的文件，并绘制参数变化曲线
    """
    def __init__(self, param_file, save_path, true_values=None, param_names=None, period=1000):
        super().__init__()
        self.param_file = param_file
        self.save_path = save_path
        self.true_values = true_values
        self.param_names = param_names
        self.period = period
        self.last_saved_step = -1

    def on_batch_end(self):
        current_step = self.model.train_state.step
        if current_step % self.period == 0 and current_step != self.last_saved_step:
            if os.path.exists(self.param_file):
                try:
                    plot_parameter_history(
                        self.param_file,
                        self.true_values,
                        self.param_names,
                        self.save_path
                    )
                except Exception:
                    pass  # 忽略绘图过程中的错误（如文件正在写入等）
            self.last_saved_step = current_step


class LossHistoryCallback(dde.callbacks.Callback):
    """
    在训练过程中实时保存loss可视化图像的Callback
    
    这个callback会在每次测试步骤（通常是display_every的倍数）时
    自动更新并保存loss历史图像，实现实时可视化。
    
    参数:
        save_dir: 保存目录
        period: 每隔多少个step保存一次（默认与display_every一致，会自动从训练参数中获取）
        filename: 保存的文件名（默认"损失历史详细图.png"）
        num_pde_losses: PDE损失的数量（用于分离PDE Loss和BC Loss）
        num_bc_losses: BC损失的数量（用于分离PDE Loss和BC Loss）
    """
    
    def __init__(self, save_dir, period=None, filename="损失历史详细图.png", 
                 num_pde_losses=None, num_bc_losses=None,
                 pde_loss_names=None, bc_loss_names=None,
                 pde_label="PDE Loss", bc_label="BC Loss",
                 save_all_components=True):
        super().__init__()
        self.save_dir = save_dir
        self.period = period  # 如果为None，会在on_train_begin中从display_every获取
        self.filename = filename
        self.num_pde_losses = num_pde_losses
        self.num_bc_losses = num_bc_losses
        self.pde_loss_names = pde_loss_names
        self.bc_loss_names = bc_loss_names
        self.pde_label = pde_label
        self.bc_label = bc_label
        self.save_all_components = save_all_components  # 是否保存所有损失项的详细图
        self.last_saved_step = -1
        
    def on_train_begin(self):
        """训练开始时初始化"""
        os.makedirs(self.save_dir, exist_ok=True)
        # 如果period未指定，尝试从训练参数中获取display_every
        # 注意：DeepXDE的callback系统不直接提供display_every，所以需要手动设置
        # 如果period为None，默认使用1000（DeepXDE的默认display_every）
        if self.period is None:
            self.period = 1000  # DeepXDE的默认display_every
        # 保存初始状态（step=0）
        self._save_loss_history()
    
    def on_batch_end(self):
        """每个batch结束时检查是否需要保存"""
        # 在_test()之后，loss history已经更新
        # 只在测试步骤（display_every的倍数）时保存
        current_step = self.model.train_state.step
        if current_step % self.period == 0 and current_step != self.last_saved_step:
            self._save_loss_history()
            self.last_saved_step = current_step
    
    def on_train_end(self):
        """训练结束时保存最终版本"""
        self._save_loss_history()
    
    def _save_loss_history(self):
        """保存loss历史图像（使用详细的损失历史图）"""
        try:
            # 1. 保存汇总图：PDE Loss、BC Loss、总Loss
            plot_and_save_loss_history(
                self.model.losshistory, 
                self.save_dir,
                filename=self.filename,
                num_pde_losses=self.num_pde_losses,
                num_bc_losses=self.num_bc_losses,
                pde_label=self.pde_label,
                bc_label=self.bc_label,
                bc_loss_names=self.bc_loss_names,
                data_loss_prefix="obs_"
            )
            
            # 2. 保存损失历史数据为JSON
            save_loss_history_json(self.model.losshistory, self.save_dir, filename="loss_history.json")
            save_best_test_loss_json(self.model.losshistory, self.save_dir, filename="best_test_loss.json")
            
            # 3. 如果需要，保存每个损失分量的详细图
            if self.save_all_components:
                plot_all_loss_components(
                    self.model.losshistory,
                    self.save_dir,
                    filename="所有损失项详细图.png",
                    num_pde_losses=self.num_pde_losses,
                    num_bc_losses=self.num_bc_losses,
                    pde_loss_names=self.pde_loss_names,
                    bc_loss_names=self.bc_loss_names
                )
            # 静默保存，不打印信息，避免输出过多（函数内部会打印）
        except Exception as e:
            # 静默处理错误，避免中断训练
            # 只在调试时打印错误信息
            import sys
            if hasattr(sys, '_getframe'):
                pass  # 静默处理
