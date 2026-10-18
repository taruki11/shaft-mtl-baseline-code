import os
import numpy as np

def load_npz_array(path: str):
    """
    支持两种格式：
      - .npy : 使用 mmap_mode='r'，只在切片时读取需要的数据（推荐）
      - .npz : 传统一次性解压读取（可能触发内存问题）
    返回形状 [C, T] 的 float32（.npy 返回 memmap / view）
    """
    path = str(path)
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        # ✅ memmap：不会把整段数据读进内存
        arr = np.load(path, mmap_mode="r")
        # 保证是 [C,T]
        if arr.ndim == 1:
            # 如果你存的是扁平一维，这里无法自动恢复 [C,T]，需要你转换时确保保存为2D
            raise ValueError(f".npy array is 1D, expected 2D [C,T]. path={path}")
        return arr

    # 默认：npz
    with np.load(path, allow_pickle=False) as z:
        # 常见键：arr_0；也兼容只有一个键的情况
        if "arr_0" in z.files:
            k = "arr_0"
        elif len(z.files) == 1:
            k = z.files[0]
        else:
            # 你也可以在这里改成你自己固定的键名
            k = z.files[0]
        arr = z[k].astype(np.float32, copy=False)

    # 保证二维 [C,T]
    if arr.ndim == 1:
        # 如果原 npz 存的是扁平一维，这里你原工程一般会 reshape
        # 但我这里不知道你的 C/T，不能瞎猜；建议转换脚本里确保保存为 2D
        raise ValueError(f".npz loaded array is 1D, expected 2D [C,T]. path={path}")
    return arr
