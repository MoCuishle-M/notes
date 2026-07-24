"""
02_channels_last.py
===================
§3.1 memory format 转换产生 non-contiguous tensor。
这是 FSDP 官方测试 test_raise_noncontiguous_parameter 用的场景。

运行: python 02_channels_last.py
"""

import torch
import torch.nn as nn


def main() -> None:
    print("=== 1. 普通 Conv2d 权重（contiguous） ===")
    conv = nn.Conv2d(8, 16, 3)
    print(f"  weight shape={tuple(conv.weight.shape)} stride={tuple(conv.weight.stride())}")
    print(f"  is_contiguous()={conv.weight.is_contiguous()}")

    print("\n=== 2. 转成 channels_last 后 ===")
    conv_cl = nn.Conv2d(8, 16, 3).to(memory_format=torch.channels_last)
    w = conv_cl.weight
    print(f"  weight shape={tuple(w.shape)} stride={tuple(w.stride())}")
    print(f"  is_contiguous()                                = {w.is_contiguous()}")
    print(f"  is_contiguous(memory_format=channels_last)     = "
          f"{w.is_contiguous(memory_format=torch.channels_last)}")
    print(f"  is_contiguous(memory_format=contiguous_format) = "
          f"{w.is_contiguous(memory_format=torch.contiguous_format)}")

    print("\n=== 3. 4D tensor 的 channels_last stride 规律 ===")
    # 标准 (N,C,H,W) contiguous stride: (C*H*W, H*W, W, 1)
    # channels_last stride:              (C*H*W, 1, C*W, C)
    print(f"  标准 row-major stride:  (C*H*W, H*W, W, 1) = "
          f"({8*5*5}, {5*5}, {5}, {1})")
    print(f"  channels_last stride:   (C*H*W, 1, C*W, C) = "
          f"({8*5*5}, {1}, {8*5}, {8})")
    print(f"  实测: {tuple(w.stride())}")

    print("\n=== 4. Module.to(memory_format=...) 会改子模块参数 ===")
    model = nn.Sequential(
        nn.Conv2d(8, 16, 3),
        nn.Conv2d(16, 32, 3),
    )
    model = model.to(memory_format=torch.channels_last)
    bad = [(n, p.is_contiguous()) for n, p in model.named_parameters()]
    print("  各参数 is_contiguous():")
    for n, c in bad:
        print(f"    {n}: {c}")

    print("\n=== 5. 退回 contiguous_format（FSDP 推荐修复方式） ===")
    conv_fixed = conv_cl.to(memory_format=torch.contiguous_format)
    print(f"  修复后 is_contiguous()={conv_fixed.weight.is_contiguous()}")


if __name__ == "__main__":
    main()
