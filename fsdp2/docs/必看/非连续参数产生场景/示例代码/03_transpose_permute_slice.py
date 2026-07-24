"""
03_transpose_permute_slice.py
============================
§3.2 / §3.3 transpose / permute / strided slicing / narrow 产生 non-contiguous tensor。

运行: python 03_transpose_permute_slice.py
"""

import torch


def show(name: str, t: torch.Tensor) -> None:
    print(f"  {name:35s} shape={tuple(t.shape):<12} stride={str(tuple(t.stride())):<16} "
          f"is_contiguous={t.is_contiguous()}")


def main() -> None:
    x = torch.randn(4, 6, 8)
    print("基础 tensor:")
    show("x = randn(4,6,8)", x)

    print("\n=== transpose / T / t() ===")
    show("x.transpose(0, 1)", x.transpose(0, 1))
    show("x.transpose(1, 2)", x.transpose(1, 2))
    show("x.T", x.T)
    y2 = torch.randn(3, 4)
    show("y2.t()  (2D)", y2.t())

    print("\n=== permute ===")
    show("x.permute(2, 0, 1)", x.permute(2, 0, 1))
    show("x.permute(0, 1, 2)  # 恒等", x.permute(0, 1, 2))

    print("\n=== strided slicing（带步长） ===")
    show("x[:, ::2, :]", x[:, ::2, :])
    show("x[:, :, ::2]", x[:, :, ::2])
    show("x[::2, ::2, :]", x[::2, ::2, :])

    print("\n=== 非 0 维范围切片（步长 1，但不在第 0 维） ===")
    show("x[:, 1:, :]", x[:, 1:, :])
    show("x[:, :, 1:]", x[:, :, 1:])
    show("x.narrow(1, 1, 4)  # 等价 x[:,1:5,:]", x.narrow(1, 1, 4))

    print("\n=== 第 0 维范围切片 → 仍连续 ===")
    show("x[1:, :, :]", x[1:, :, :])
    show("x[:2, :, :]", x[:2, :, :])
    show("x.narrow(0, 1, 2)", x.narrow(0, 1, 2))

    print("\n=== contiguous() 修复 ===")
    t = x.transpose(0, 1)
    c = t.contiguous()
    show("x.transpose(0,1)", t)
    show("x.transpose(0,1).contiguous()", c)
    print(f"  共享 storage? {t.data_ptr() == c.data_ptr()}  (False=发生了拷贝)")


if __name__ == "__main__":
    main()
