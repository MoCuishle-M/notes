"""
05_non_contiguous_parameter_scenarios.py
========================================
§4 实际场景：哪些写法会让 nn.Parameter 本身变成 non-contiguous。
nn.Parameter 内部不会自动 .contiguous()，所以传入什么就是什么。

运行: python 05_non_contiguous_parameter_scenarios.py
"""

import torch
import torch.nn as nn


def report(name: str, p: nn.Parameter) -> None:
    flag = "OK " if p.is_contiguous() else "BAD"
    print(f"  [{flag}] {name:38s} shape={tuple(p.shape)} stride={tuple(p.stride())}")


def main() -> None:
    print("=== 1. nn.Linear 默认权重（contiguous，FSDP 接受） ===")
    lin = nn.Linear(10, 5)
    report("linear.weight", lin.weight)
    report("linear.bias",   lin.bias)

    print("\n=== 2. Conv2d + channels_last（FSDP 测试用例原型，BAD） ===")
    conv = nn.Conv2d(8, 16, 3).to(memory_format=torch.channels_last)
    report("conv.weight (channels_last)", conv.weight)

    print("\n=== 3. 手动把 .T 当参数（BAD） ===")
    lin2 = nn.Linear(10, 5)
    bad_T = nn.Parameter(lin2.weight.T)                # 转置 view 直接当参数
    good_T = nn.Parameter(lin2.weight.T.contiguous())  # 修复
    report("nn.Parameter(weight.T)", bad_T)
    report("nn.Parameter(weight.T.contiguous())", good_T)

    print("\n=== 4. permute 后当参数（BAD） ===")
    w = torch.randn(4, 6, 8)
    bad_perm = nn.Parameter(w.permute(2, 0, 1))
    report("nn.Parameter(w.permute(2,0,1))", bad_perm)

    print("\n=== 5. 从大参数切片（看切哪一维） ===")
    big = nn.Parameter(torch.randn(10, 20))
    good_slice_0 = nn.Parameter(big[:5, :])        # 第 0 维切片 → 仍连续
    bad_slice_1  = nn.Parameter(big[:, :10])       # 非 0 维切片 → 非连续
    report("nn.Parameter(big[:5, :])",   good_slice_0)
    report("nn.Parameter(big[:, :10])",  bad_slice_1)

    print("\n=== 6. expand 用于 tied / broadcast 权重（BAD） ===")
    base = nn.Parameter(torch.randn(1, 8, 4))
    bad_expand = nn.Parameter(base.expand(3, 8, 4))
    report("nn.Parameter(base.expand(3,8,4))", bad_expand)

    print("\n=== 7. flip 后当参数（OK，2.12 返回 contiguous copy） ===")
    base2 = torch.randn(3, 4)
    ok_flip = nn.Parameter(torch.flip(base2, dims=[0]))
    report("nn.Parameter(torch.flip(base2,[0]))", ok_flip)

    print("\n=== 8. as_strided 后当参数（BAD） ===")
    storage = torch.randn(24)
    bad_strided = nn.Parameter(torch.as_strided(storage, (3, 4), (8, 2)))
    report("nn.Parameter(as_strided(...))", bad_strided)

    print("\n=== 9. diagonal 后当参数（BAD） ===")
    m = torch.randn(4, 4)
    bad_diag = nn.Parameter(torch.diagonal(m, offset=1))
    report("nn.Parameter(torch.diagonal(m,1))", bad_diag)

    print("\n=== 10. .to(device) 不会改变 contiguity（preserve_format） ===")
    # CPU 上用 .to('cpu') 演示，逻辑与 .cuda() 一致
    conv_cl = nn.Conv2d(8, 16, 3).to(memory_format=torch.channels_last)
    conv_moved = conv_cl.to(device='cpu')
    report("channels_last conv (before .to)", conv_cl.weight)
    report("channels_last conv (after  .to)", conv_moved.weight)

    print("\n=== 11. 体检函数：fully_shard 之前的预检 ===")
    def check_contiguous(module: nn.Module) -> None:
        bad = []
        for name, p in module.named_parameters():
            if not p.is_contiguous():
                bad.append((name, tuple(p.shape), tuple(p.stride())))
        if bad:
            print(f"  [FAIL] 发现 {len(bad)} 个 non-contiguous 参数:")
            for n, s, st in bad:
                print(f"     {n}: shape={s} stride={st}")
        else:
            print("  [PASS] 所有参数都是 contiguous")

    model = nn.Sequential(
        nn.Conv2d(8, 16, 3),
        nn.Conv2d(16, 32, 3),
    ).to(memory_format=torch.channels_last)
    check_contiguous(model)

    model_fixed = model.to(memory_format=torch.contiguous_format)
    check_contiguous(model_fixed)


if __name__ == "__main__":
    main()
