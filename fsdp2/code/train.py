import os
import argparse
import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh
from torch.distributed.fsdp import FSDPModule
from torch.nn.parallel import DistributedDataParallel as DDP

from models import MultiModalModel
from fsdp_utils import apply_fsdp2

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed-backend", type=str, default="gloo",
                        help="分布式后端，CPU 环境下强制使用 gloo")
    return parser.parse_args()


def main():
    args = parse_args()

    # 强制使用 CPU
    device = torch.device("cpu")
    torch.set_default_device(device)

    # 始终初始化进程组（torchrun 会设置 RANK 环境变量）
    if "RANK" in os.environ:
        backend = "gloo"  # CPU 只能使用 gloo
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"[Rank {rank}] World size: {world_size}, backend: {backend}, device: {device}")
    else:
        rank = 0
        world_size = 1
        print(f"Running in single-process debug mode (no distributed), device: {device}")

    model = MultiModalModel(
        image_size=64, patch_size=16, in_channels=3,
        vit_embed_dim=128, vit_depth=20, vit_heads=4,
        vocab_size=1000, llm_dim=128, llm_depth=10, llm_heads=4,
        num_experts=36, top_k=8, max_seq_len=128
    )
    print(f"[Rank {rank}] Model param count: {sum(p.numel() for p in model.parameters()):,}")

    if world_size > 1:
        mesh = DeviceMesh("cpu", torch.arange(world_size).tolist())
        fsdp_model = apply_fsdp2(model, mesh)
        print(f"[Rank {rank}] FSDP2 applied with mesh size {mesh.size()}")
    else:
        fsdp_model = DDP(model, device_ids=None)
        print(f"[Rank {rank}] Using ordinary DDP")

    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-3)

    batch_size = 2
    seq_len = 32
    images = torch.randn(batch_size, 3, 64, 64, device=device)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    labels = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    fsdp_model.train()
    for step in range(5):
        optimizer.zero_grad()
        logits, loss = fsdp_model(images, input_ids, labels=labels)
        print(f"[Rank {rank}] Step {step}: loss = {loss.item():.4f}")
        loss.backward()
        optimizer.step()

    # 打印 FSDP 分片信息（单卡也能看到 FSDPModule 包装）
    if isinstance(fsdp_model, FSDPModule):
        print(f"\n[Rank {rank}] --- FSDP State Debug ---")
        for name, param in fsdp_model.named_parameters():
            if hasattr(param, '_local_tensor'):
                local_shape = param._local_tensor.shape
                global_shape = param.shape
                print(f"  [Rank {rank}] {name}: global={global_shape}, local={local_shape}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
