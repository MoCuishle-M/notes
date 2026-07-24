"""
06_fully_shard_rejection_demo.py
================================
§2 FSDP2 拒绝 non-contiguous 参数的端到端演示。

注意：需要 CUDA + 多进程分布式环境。在 CPU 单进程下无法真正调用 fully_shard，
本脚本分两部分：
  (A) 复现官方测试 test_raise_noncontiguous_parameter 的构造方式；
  (B) 给出在真实分布式环境下的运行命令与最小示例（注释形式）。

运行方式（分布式，2 GPU）:
    torchrun --nproc_per_node=2 06_fully_shard_rejection_demo.py
"""

import torch
import torch.nn as nn


def part_a_construct_non_contiguous_param() -> None:
    """复现官方测试 test_raise_noncontiguous_parameter 的构造。"""
    print("=== Part A: 构造 non-contiguous 参数（CPU 即可） ===")
    conv2d = nn.Conv2d(8, 8, 3).to(memory_format=torch.channels_last)
    print(f"  conv2d.weight.shape    = {tuple(conv2d.weight.shape)}")
    print(f"  conv2d.weight.stride   = {tuple(conv2d.weight.stride())}")
    print(f"  is_contiguous()        = {conv2d.weight.is_contiguous()}")
    print(f"  is_contiguous(memory_format=channels_last) = "
          f"{conv2d.weight.is_contiguous(memory_format=torch.channels_last)}")
    print("  -> 这样的 conv2d 传给 fully_shard 会抛 NotImplementedError")


def part_b_distributed_demo() -> None:
    """
    真实分布式环境下复现 FSDP 拒绝逻辑的最小示例。
    以下代码仅在 rank 0 执行打印，但 fully_shard 在所有 rank 上都会抛错。
    """
    if not torch.distributed.is_available():
        print("\n=== Part B: 跳过（torch.distributed 不可用） ===")
        return

    # 仅当通过 torchrun 启动时才初始化
    if not torch.distributed.is_initialized():
        try:
            torch.distributed.init_process_group(backend="nccl")
        except Exception as e:
            print(f"\n=== Part B: 跳过分布式初始化（{e}） ===")
            return

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if rank == 0:
        print(f"\n=== Part B: 分布式复现 (world_size={world_size}) ===")

    assert torch.cuda.is_available(), "需要 CUDA 环境"
    torch.cuda.set_device(rank)

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    mesh = init_device_mesh("cuda", (world_size,))

    # ---- 场景 1: channels_last conv -> 应被拒绝 ----
    conv_bad = nn.Conv2d(8, 16, 3).to(memory_format=torch.channels_last).cuda()
    try:
        fully_shard(conv_bad, mesh=mesh)
        if rank == 0:
            print("  [UNEXPECTED] 期望抛错但未抛错（FSDP 行为变化?）")
    except NotImplementedError as e:
        if rank == 0:
            print(f"  [REJECTED] 场景1 被拒绝: {e}")

    # ---- 场景 2: 修复后 -> 应通过 ----
    conv_good = nn.Conv2d(8, 16, 3).cuda()  # 默认 contiguous_format
    try:
        fully_shard(conv_good, mesh=mesh)
        if rank == 0:
            print(f"  [ACCEPTED] 场景2 通过: contiguous conv 已被 fully_shard 接受")
    except Exception as e:
        if rank == 0:
            print(f"  [ERROR] 场景2 异常: {type(e).__name__}: {e}")

    # ---- 场景 3: 手动 transposed 权重 -> 应被拒绝 ----
    lin_bad = nn.Linear(10, 5).cuda()
    lin_bad.weight = nn.Parameter(lin_bad.weight.T.cuda())  # non-contiguous
    try:
        fully_shard(lin_bad, mesh=mesh)
        if rank == 0:
            print("  [UNEXPECTED] 期望抛错但未抛错（FSDP 行为变化?）")
    except NotImplementedError as e:
        if rank == 0:
            print(f"  [REJECTED] 场景3 被拒绝: {e}")


if __name__ == "__main__":
    part_a_construct_non_contiguous_param()
    part_b_distributed_demo()
