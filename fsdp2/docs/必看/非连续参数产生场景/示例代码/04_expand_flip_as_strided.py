"""
04_expand_flip_as_strided.py
============================
§3.4 / §3.5 / §3.6 / §3.7 expand / flip / as_strided / diagonal 的 contiguity 实测。

实测结论（PyTorch 2.12）：
  - expand / as_strided / diagonal → 非连续（view）
  - flip / triu / tril              → 连续拷贝（不是 view，不会产生 non-contiguous 参数）

运行: python 04_expand_flip_as_strided.py
"""

import torch


def show(name: str, t: torch.Tensor) -> None:
    print(f"  {name:40s} shape={str(tuple(t.shape)):<12} stride={str(tuple(t.stride())):<16} "
          f"is_contiguous={t.is_contiguous()}")


def main() -> None:
    print("=== 1. expand / expand_as → 非连续 ===")
    a = torch.randn(1, 5)
    show("a = randn(1,5)", a)
    show("a.expand(3, 5)", a.expand(3, 5))
    show("a.expand(4, 3, 5)", a.expand(4, 3, 5))

    b = torch.randn(2, 1, 4)
    show("b = randn(2,1,4)", b)
    show("b.expand(2, 6, 4)", b.expand(2, 6, 4))

    print("\n=== 2. flip → 实测返回 contiguous copy（不是非连续 view） ===")
    x = torch.randn(3, 4)
    show("x = randn(3,4)", x)
    f0 = torch.flip(x, dims=[0])
    f1 = torch.flip(x, dims=[1])
    f01 = torch.flip(x, dims=[0, 1])
    show("torch.flip(x, dims=[0])", f0)
    show("torch.flip(x, dims=[1])", f1)
    show("torch.flip(x, dims=[0,1])", f01)
    # 用 data_ptr 判断是否拷贝：与 x 不同说明是 copy
    print(f"  flip(x,[0]) data_ptr == x.data_ptr()? {f0.data_ptr() == x.data_ptr()}  "
          f"(False=发生了拷贝)")
    print(f"  内容翻转验证 f0[0] == x[2]? {torch.equal(f0[0], x[2])}")

    print("\n=== 3. as_strided（可任意指定 stride）→ 通常非连续 ===")
    storage = torch.randn(24)
    print(f"  storage len={len(storage.untyped_storage())}")
    show("as_strided((3,4),(8,2))", torch.as_strided(storage, (3, 4), (8, 2)))
    show("as_strided((3,4),(4,1))  # 标准", torch.as_strided(storage, (3, 4), (4, 1)))
    show("as_strided((2,3,4),(12,4,1))  # 标准 3D", torch.as_strided(storage, (2, 3, 4), (12, 4, 1)))

    print("\n=== 4. diagonal → 非连续；triu / tril → contiguous copy ===")
    m = torch.randn(4, 4)
    show("torch.diagonal(m)", torch.diagonal(m))
    show("torch.diagonal(m, offset=1)", torch.diagonal(m, offset=1))
    show("torch.triu(m, diagonal=1)", torch.triu(m, diagonal=1))
    show("torch.tril(m, diagonal=-1)", torch.tril(m, diagonal=-1))
    # 验证 triu 是 copy
    tu = torch.triu(m, diagonal=1)
    print(f"  triu data_ptr == m.data_ptr()? {tu.data_ptr() == m.data_ptr()}  "
          f"(False=发生了拷贝)")

    print("\n=== 5. unfold → 非连续 ===")
    u = torch.randn(2, 8)
    show("u = randn(2,8)", u)
    show("u.unfold(1, 3, 2)", u.unfold(1, 3, 2))

    print("\n=== 6. 修复：统一 .contiguous() ===")
    cases = {
        "expand":    a.expand(3, 5),
        "as_strided": torch.as_strided(storage, (3, 4), (8, 2)),
        "diagonal":  torch.diagonal(m, offset=1),
        "unfold":    u.unfold(1, 3, 2),
    }
    for name, t in cases.items():
        c = t.contiguous()
        print(f"  {name:12s} 原始 contiguous={t.is_contiguous()}  "
              f"修复后 contiguous={c.is_contiguous()}")

    print("\n=== 7. 小结 ===")
    print("  非连续（view，FSDP 风险）: expand, as_strided, diagonal, unfold")
    print("  连续拷贝（安全）        : flip, triu, tril")


if __name__ == "__main__":
    main()
