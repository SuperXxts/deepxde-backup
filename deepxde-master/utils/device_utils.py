"""
设备相关的工具函数
用于检查GPU/CPU等设备信息，以及设置随机种子
"""
import os
import deepxde as dde


def check_pytorch_gpu():
    """
    检查PyTorch的GPU使用情况
    
    返回:
        dict: 包含GPU信息的字典
            - available: bool, GPU是否可用
            - device_count: int, GPU数量
            - device_name: str, GPU名称
            - default_device: str, 默认设备
    """
    try:
        import torch
        
        gpu_info = {
            "available": torch.cuda.is_available(),
            "device_count": 0,
            "device_name": None,
            "default_device": "cpu"
        }
        
        if gpu_info["available"]:
            gpu_info["device_count"] = torch.cuda.device_count()
            if gpu_info["device_count"] > 0:
                gpu_info["device_name"] = torch.cuda.get_device_name(0)
            if hasattr(torch, 'get_default_device'):
                gpu_info["default_device"] = str(torch.get_default_device())
            else:
                gpu_info["default_device"] = "cuda"
        
        return gpu_info
    except ImportError:
        return {
            "available": False,
            "device_count": 0,
            "device_name": None,
            "default_device": "cpu",
            "error": "PyTorch not installed"
        }


def print_gpu_info():
    """
    打印GPU使用情况信息（仅用于PyTorch后端）
    """
    print("=" * 60)
    print("GPU/设备信息:")
    print(f"  后端 (Backend): {dde.backend.backend_name}")
    
    if dde.backend.backend_name == "pytorch":
        gpu_info = check_pytorch_gpu()
        if gpu_info["available"]:
            print(f"  GPU可用: 是")
            print(f"  GPU数量: {gpu_info['device_count']}")
            if gpu_info["device_name"]:
                print(f"  当前GPU: {gpu_info['device_name']}")
            print(f"  默认设备: {gpu_info['default_device']}")
        else:
            print(f"  GPU可用: 否 (使用CPU)")
            if "error" in gpu_info:
                print(f"  错误: {gpu_info['error']}")
    else:
        print(f"  注意: 当前后端为 {dde.backend.backend_name}，此函数仅检查PyTorch GPU")
    
    print("=" * 60)
    print()


def set_random_seed(seed):
    """
    设置随机种子，确保实验结果可复现
    
    这个函数会设置所有相关的随机数生成器的种子：
    - Python random 模块
    - NumPy 随机数生成器
    - 后端框架（PyTorch/TensorFlow等）的随机数生成器
    
    注意：固定seed可能会略微降低训练速度（启用确定性操作），
    但能保证结果的一致性，便于实验复现和对比。
    
    参数:
        seed (int or None): 随机种子值
            - 如果为整数：设置随机种子，结果可复现
            - 如果为None：不设置随机种子，结果不可复现
    
    返回:
        int or None: 返回设置的种子值
    """
    if seed is not None:
        dde.config.set_random_seed(seed)
        
        # 针对 PyTorch 后端的额外确定性设置
        if dde.backend.backend_name == "pytorch":
            import torch
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(True)
            # 设置所有 GPU 的随机种子
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                # 确保卷积操作等是确定性的
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                print(f"PyTorch GPU 随机种子已设置，并启用了确定性算法")
                
        print(f"随机种子已设置为: {seed} (结果可复现)")
        print()
    else:
        print("未设置随机种子 (结果不可复现)")
        print()
    return seed
