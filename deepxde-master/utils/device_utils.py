"""
设备相关的工具函数
用于检查GPU/CPU等设备信息，以及设置随机种子
"""
import json
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


def enforce_and_report_runtime_device(net=None, save_dir=None, require_cuda=True):
    """
    显式绑定 PyTorch/DCU 设备，并将运行时设备信息写入日志/JSON。
    """
    info = {
        "backend": dde.backend.backend_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "rocm_path": os.environ.get("ROCM_PATH"),
        "hip_path": os.environ.get("HIP_PATH"),
        "cuda_available": False,
        "device_count": 0,
        "device_name": None,
        "runtime_probe_device": None,
        "first_param_device": None,
        "first_buffer_device": None,
        "memory_allocated": None,
    }

    if dde.backend.backend_name != "pytorch":
        if save_dir is not None:
            os.makedirs(os.path.join(save_dir, "json"), exist_ok=True)
            with open(os.path.join(save_dir, "json", "device_runtime.json"), "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
        return info

    import torch

    info["cuda_available"] = bool(torch.cuda.is_available())
    info["device_count"] = int(torch.cuda.device_count()) if info["cuda_available"] else 0
    if info["cuda_available"] and info["device_count"] > 0:
        torch.cuda.set_device(0)
        info["device_name"] = torch.cuda.get_device_name(0)
        probe = torch.zeros((1,), device="cuda")
        info["runtime_probe_device"] = str(probe.device)
        if net is not None and hasattr(net, "to"):
            net.to("cuda")
        if net is not None:
            try:
                first_param = next(net.parameters())
                info["first_param_device"] = str(first_param.device)
            except StopIteration:
                info["first_param_device"] = None
            try:
                first_buffer = next(net.buffers())
                info["first_buffer_device"] = str(first_buffer.device)
            except StopIteration:
                info["first_buffer_device"] = None
        info["memory_allocated"] = int(torch.cuda.memory_allocated())
    elif require_cuda:
        raise RuntimeError("torch.cuda.is_available() is False. Current run is not using DCU/GPU.")

    print("=" * 60)
    print("Runtime Device Check:")
    for key in [
        "backend",
        "cuda_visible_devices",
        "rocr_visible_devices",
        "hip_visible_devices",
        "cuda_available",
        "device_count",
        "device_name",
        "runtime_probe_device",
        "first_param_device",
        "first_buffer_device",
        "memory_allocated",
    ]:
        print(f"  {key}: {info[key]}")
    print("=" * 60)
    print()

    if require_cuda and info["cuda_available"]:
        param_device = info.get("first_param_device")
        if param_device is None or not param_device.startswith("cuda"):
            raise RuntimeError(f"Model parameters are not on DCU/GPU. first_param_device={param_device}")

    if save_dir is not None:
        os.makedirs(os.path.join(save_dir, "json"), exist_ok=True)
        with open(os.path.join(save_dir, "json", "device_runtime.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
    return info


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
