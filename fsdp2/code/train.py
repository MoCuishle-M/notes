import os
import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh
from torch.distributed.fsdp._fully_shard import FullyShardedDataParallel

from models import MultiModalModel
from fsdp_utils import apply_fsdp2

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"


def main():
    if "RANK" in os.environ:
        dist.init_process_group(backend="gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device("cpu")
        torch.set_default_device(device)
        print(f"[Rank {rank}] World size: {world_size}, backend: gloo")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cpu")
        torch.set_default_device(device)
        print("Running in single-process debug mode (no distributed)")

    model = MultiModalModel(
        image_size=64, patch_size=16, in_channels=3,
        vit_embed_dim=128, vit_depth=2, vit_heads=4,
        vocab_size=1000, llm_dim=128, llm_depth=2, llm_heads=4,
        num_experts=2, top_k=1, max_seq_len=128
    )
    print(f"Model param count: {sum(p.numel() for p in model.parameters()):,}")

    if world_size > 1:
        mesh = DeviceMesh("cpu", torch.arange(world_size).tolist())
        fsdp_model = apply_fsdp2(model, mesh)
    else:
        fsdp_model = model

    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-3)

    batch_size = 2
    seq_len = 32
    images = torch.randn(batch_size, 3, 64, 64)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = torch.randint(0, 1000, (batch_size, seq_len))

    fsdp_model.train()
    for step in range(5):
        optimizer.zero_grad()
        logits, loss = fsdp_model(images, input_ids, labels=labels)
        print(f"[Rank {rank}] Step {step}: loss = {loss.item():.4f}")
        loss.backward()
        optimizer.step()

    if world_size > 1 and isinstance(fsdp_model, FullyShardedDataParallel):
        #print("\n--- FSDP State Debug ---")
        for name, param in fsdp_model.named_parameters():
            if hasattr(param, '_local_tensor'):
                local_shape = param._local_tensor.shape
                global_shape = param.shape
                print(f"  {name}: global={global_shape}, local={local_shape}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
