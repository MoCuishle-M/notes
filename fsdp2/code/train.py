import os
import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh
from torch.distributed.fsdp import FSDPModule

from models import MultiModalModel
from fsdp_utils import apply_fsdp2

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.set_default_device(device)

    if "RANK" in os.environ:
        backend = "nccl" if use_cuda else "gloo"
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
    print(f"Model param count: {sum(p.numel() for p in model.parameters()):,}")

    if world_size > 1:
        mesh = DeviceMesh(device.type, torch.arange(world_size).tolist())
        fsdp_model = apply_fsdp2(model, mesh)
    else:
        fsdp_model = model

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

    if world_size > 1 and isinstance(fsdp_model, FSDPModule):
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
