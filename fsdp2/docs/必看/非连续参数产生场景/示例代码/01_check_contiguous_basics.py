"""
01_check_contiguous_basics.py
=============================
演示如何判断 tensor 是否连续、stride 与 contiguity 的关系。

运行: python 01_check_contiguous_basics.py
"""

import torch


def show(name: str, t: torch.Tensor) -> None:
    print(f"{name:30s} shape={tuple(t.shape)} stride={tuple(t.stride())} "
          f"is_contiguous={t.is_contiguous()}")


def main() -> None:
    print("=== 1. 标准连续 tensor ===")
    a = torch.randn(2, 3, 4)
    show("torch.randn(2,3,4)", a)

    print("\n=== 2. 第 0 维切片 → 仍然连续 ===")
    show("a[1:, :]", a[1:, :])
    show("a[:1, :]", a[:1, :])

    print("\n=== 3. 非 0 维切片 → 非连续 ===")
    show("a[:, 1:, :]", a[:, 1:, :])
    show("a[:, :, ::2]", a[:, :, ::2])

    print("\n=== 4. transpose / permute → 非连续 ===")
    show("a.transpose(0, 1)", a.transpose(0, 1))
    show("a.permute(2, 0, 1)", a.permute(2, 0, 1))

    print("\n=== 5. contiguous() 显式拷贝 ===")
    t = a.transpose(0, 1)
    c = t.contiguous()
    show("a.transpose(0,1)", t)
    show("a.transpose(0,1).contiguous()", c)
    print(f"  data_ptr 相同? {t.data_ptr() == c.data_ptr()}  (False 说明发生了拷贝)")

    print("\n=== 6. storage / offset 概念 ===")
    s = torch.randn(12)
    print(f"storage len={len(s.untyped_storage())}")
    view1 = s.view(3, 4)
    view2 = s.as_strided((3, 4), (4, 1))
    show("s.view(3,4)", view1)
    show("s.as_strided((3,4),(4,1))", view2)
    print(f"  view1.data_ptr()==view2.data_ptr()? {view1.data_ptr() == view2.data_ptr()}")

    print("\n=== 7. 用 is_contiguous(memory_format=...) 判断 channels_last ===")
    # 注意：memory_format 必须用关键字传：is_contiguous(memory_format=...)
    # 不能写成 is_contiguous(torch.channels_last)（2.12 起仅接受关键字参数）。
    x = torch.randn(2, 3, 4, 4)
    cl = x.to(memory_format=torch.channels_last)
    print(f"channels_last tensor:")
    print(f"  is_contiguous()                                = {cl.is_contiguous()}")
    print(f"  is_contiguous(memory_format=channels_last)     = "
          f"{cl.is_contiguous(memory_format=torch.channels_last)}")
    print(f"  is_contiguous(memory_format=contiguous_format) = "
          f"{cl.is_contiguous(memory_format=torch.contiguous_format)}")


if __name__ == "__main__":
    main()
