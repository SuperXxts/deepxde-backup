"""
权重保存和加载工具函数
"""
import os
import glob
import json
import numpy as np
from datetime import datetime
import deepxde as dde


class BestModelCheckpoint(dde.callbacks.Callback):
    """
    自定义callback，只保留一个最优权重文件
    每次保存新的最优权重时，自动删除旧的最优权重文件
    同时保存对应的详细loss信息到json文件
    """
    def __init__(self, filepath, monitor="test loss", period=1, verbose=1):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.period = period
        self.verbose = verbose
        self.monitor_op = np.less
        self.epochs_since_last_save = 0
        self.best = np.inf
        self.best_file = None  # 记录当前最优权重文件路径
        self.best_step = 0     # 记录最优权重对应的step
    
    def on_epoch_end(self):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save < self.period:
            return
        self.epochs_since_last_save = 0
        
        current = self.get_monitor_value()
        if self.monitor_op(current, self.best):
            # 删除旧的最优权重文件
            if self.best_file and os.path.exists(self.best_file):
                try:
                    os.remove(self.best_file)
                except:
                    pass
            
            # 保存新的最优权重
            save_path = self.model.save(self.filepath, verbose=0)
            self.best_file = save_path
            self.best = current
            self.best_step = self.model.train_state.iteration
            
            # 保存详细loss信息
            self._save_best_info(save_path, current)
            
            if self.verbose > 0:
                print(
                    "Epoch {}: {} improved from {:.2e} to {:.2e}, saving model to {} ...\n".format(
                        self.model.train_state.iteration,
                        self.monitor,
                        self.best,
                        current,
                        save_path,
                    )
                )

    def _save_best_info(self, model_path, current_loss):
        """保存最优模型对应的详细loss信息"""
        try:
            # 获取保存目录
            save_dir = os.path.dirname(model_path)
            json_path = os.path.join(save_dir, "best_loss_info.json")
            
            # 准备数据
            # 注意：train_state.loss_train 是一个列表，包含各个loss分量
            loss_train = [float(x) for x in self.model.train_state.loss_train]
            loss_test = [float(x) for x in self.model.train_state.loss_test] if self.model.train_state.loss_test else []
            metrics = [float(x) for x in self.model.train_state.metrics_test] if self.model.train_state.metrics_test else []
            
            info = {
                "step": int(self.model.train_state.iteration),
                "monitor": self.monitor,
                "best_value": float(current_loss),
                "loss_train_components": loss_train,
                "loss_validation_components": loss_test,
                "metrics_validation": metrics,
                "model_path": os.path.basename(model_path),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Failed to save best loss info: {e}")
    
    def get_monitor_value(self):
        if self.monitor == "train loss":
            result = sum(self.model.train_state.loss_train)
        elif self.monitor == "test loss":
            result = sum(self.model.train_state.loss_test)
        else:
            raise ValueError(f"Unknown monitor: {self.monitor}")
        return result


def create_best_model_checkpoint(filepath, monitor="test loss", period=1, verbose=1):
    """
    创建最优权重保存callback
    
    Args:
        filepath: 权重文件路径
        monitor: 监控的loss类型，"train loss" 或 "test loss"
        period: 检查频率（每多少步检查一次）
        verbose: 是否打印保存信息
    
    Returns:
        BestModelCheckpoint callback对象
    """
    return BestModelCheckpoint(
        filepath=filepath,
        monitor=monitor,
        period=period,
        verbose=verbose
    )


def save_last_weights(model, save_dir, filename="model_last_weights.pth"):
    """
    保存最后权重（训练结束时的权重）
    自动删除所有带迭代次数的旧文件，只保留一个固定文件名的权重文件
    
    Args:
        model: DeepXDE的Model对象
        save_dir: 保存目录
        filename: 权重文件名
    
    Returns:
        saved_path: 保存的文件路径
    """
    last_model_path = os.path.join(save_dir, filename)
    try:
        saved_path = model.save(last_model_path)
        # 删除所有旧的最后权重文件（带迭代次数的）
        pattern = os.path.join(save_dir, f"{filename}-*")
        for old_file in glob.glob(pattern):
            if old_file != saved_path:
                try:
                    os.remove(old_file)
                except:
                    pass
        # 重命名为固定文件名（去掉迭代次数）
        if saved_path != last_model_path:
            if os.path.exists(last_model_path):
                os.remove(last_model_path)
            os.rename(saved_path, last_model_path)
        print(f"✓ 最后权重已保存: {last_model_path}")
        return last_model_path
    except Exception as e:
        print(f"✗ 保存最后权重失败: {e}")
        return None


def load_best_weights(model, save_dir, filename="model_best_weights.pth", verbose=1):
    """
    加载最优权重（训练过程中验证loss最小的权重）
    自动查找最新的最优权重文件，加载后删除所有旧文件并重命名为固定文件名
    
    Args:
        model: DeepXDE的Model对象
        save_dir: 保存目录
        filename: 权重文件名
        verbose: 是否打印详细信息
    
    Returns:
        best_model_path: 最优权重文件路径，如果不存在则返回None
    """
    best_model_path = os.path.join(save_dir, filename)
    # 查找最优权重文件（可能带迭代次数）
    best_files = glob.glob(os.path.join(save_dir, f"{filename}*"))
    if best_files:
        try:
            # 找到最新的最优权重文件（按修改时间）
            best_file = max(best_files, key=os.path.getmtime)
            # 加载最优权重
            model.restore(best_file, verbose=verbose)
            if verbose > 0:
                print(f"✓ 最优权重已加载: {best_file}")
            
            # 如果文件名带迭代次数，重命名为固定文件名并删除其他旧文件
            if best_file != best_model_path:
                # 删除所有旧的最优权重文件
                for old_file in best_files:
                    if old_file != best_file:
                        try:
                            os.remove(old_file)
                        except:
                            pass
                # 重命名为固定文件名
                if os.path.exists(best_model_path):
                    os.remove(best_model_path)
                os.rename(best_file, best_model_path)
                if verbose > 0:
                    print(f"  已重命名为: {best_model_path}")
            return best_model_path
        except Exception as e:
            print(f"✗ 加载最优权重失败: {e}")
            return None
    else:
        if verbose > 0:
            print(f"✗ 最优权重文件不存在: {best_model_path}")
        return None


def resolve_model_path(model_dir, save_dir, model_path=None):
    if model_path:
        return model_path
    if os.path.exists(model_dir):
        best_files = [f for f in os.listdir(model_dir) if f.startswith("best_model")]
        if best_files:
            best_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
            latest_best = best_files[-1]
            if latest_best.endswith(".index"):
                latest_best = latest_best.replace(".index", "")
            if latest_best.endswith(".pt"):
                latest_best = latest_best.replace(".pt", "")
            return os.path.join(model_dir, latest_best)
        return os.path.join(model_dir, "model")
    return os.path.join(save_dir, "model/model")
