import torch.nn as nn
from torch.distributed.fsdp import fully_shard


def apply_fsdp2(model, mesh):
    """
    使用 fully_shard 对模型的各个部分进行细粒度分片。
    采用 "逐层分片" 策略，将每个 TransformerBlock 以及 patch_embed、
    lm_head 等分别包装，以最大化通信与计算重叠。
    """
    fully_shard(model.vit.patch_embed, mesh=mesh)
    for blk in model.vit.blocks:
        fully_shard(blk, mesh=mesh)
    fully_shard(model.vit.norm, mesh=mesh)

    fully_shard(model.llm.token_embed, mesh=mesh)
    for blk in model.llm.blocks:
        fully_shard(blk, mesh=mesh)
    fully_shard(model.llm.norm, mesh=mesh)
    fully_shard(model.llm.lm_head, mesh=mesh)

    if isinstance(model.vision_proj, nn.Linear):
        fully_shard(model.vision_proj, mesh=mesh)

    fsdp_model = fully_shard(model, mesh=mesh)
    return fsdp_model
