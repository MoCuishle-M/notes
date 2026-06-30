# 示例代码与 fully_shard 行为详解

> 基于 PyTorch v2.12.0 源码分析
> [源码链接](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/distributed/fsdp/_fully_shard/_fully_shard.py)

## 1. 示例代码：创建小模型并使用 fully_shard

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

# 初始化分布式环境
dist.init_process_group(backend="nccl")
local_rank = int(dist.get_rank())

# 创建 2D 设备网格 (用于 HSDP)
# dim 0: 数据并行复制组 (replicate), dim 1: FSDP 分片组 (shard)
mesh = init_device_mesh(
    "cuda",
    mesh_shape=(2, 4),  # 共 8 GPU: 2 个复制组 × 4 个分片组
    mesh_dim_names=("dp_replicate", "dp_shard"),
)

# 定义一个小模型
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

model = SimpleMLP().cuda()

# ====== 方式 1: fully_shard(model) — 整体包装 ======
# 对整个模型应用 FSDP
fsdp_model = fully_shard(model, mesh=mesh)

# ====== 方式 2: fully_shard(model.module) — 逐层包装 ======
# 对每个子模块分别应用 FSDP（更细粒度的分片）
model2 = SimpleMLP().cuda()
for layer in model2.children():
    if isinstance(layer, nn.Linear):
        fully_shard(layer, mesh=mesh)
fully_shard(model2, mesh=mesh)  # 顶层也需要包装
```

## 2. `fully_shard(model)` 之后发生了什么

调用 `fully_shard(model)` 后，FSDP2 执行以下关键步骤：

### 2.1 模块验证与 Mesh 初始化

```python
# _fully_shard.py 中的实现
@contract(state_cls=FSDPState)
def fully_shard(module, *, mesh=None, ...):
    _validate_module(module)  # 检查是否为 nn.ModuleList/ModuleDict 等容器
    if mesh is None:
        mesh = _init_default_fully_shard_mesh()  # 创建默认 1D mesh
    _validate_mesh(mesh)
    mesh_info = _get_mesh_info(mesh)  # 解析为 FSDPMeshInfo/HSDPMeshInfo/DDPMeshInfo
```

### 2.2 MRO 替换（类层次结构修改）

这是 FSDP2 最核心的机制。FSDP2 通过 **修改模块的 `__class__`** 来拦截 `forward` 调用：

```python
# _fully_shard.py: _apply_to_module()
def _apply_to_module(modules, cls_to_wrapper_cls, wrapper_module_cls, ...):
    for module in modules:
        cls = module.__class__
        new_cls = type(
            f"FSDP{cls.__name__}",  # 类名：FSDPSimpleMLP
            (FSDPModule, cls),      # 继承自 FSDPModule（左）和原类（右）
            {"__deepcopy__": _unimplemented_deepcopy}
        )
        module.__class__ = new_cls  # 直接修改 class！
```

修改后的 MRO（方法解析顺序）变为：

```text
FSDPSimpleMLP -> FSDPModule -> SimpleMLP -> nn.Module -> ...
```

由于 `FSDPModule` 在原始类 `SimpleMLP` 之前，`FSDPModule.forward` 会先被调用。

### 2.3 `FSDPModule.__new__`拦截实例化

```python
# _fully_shard.py: FSDPModule.__new__
class FSDPModule:
    def __new__(cls, *args, **kwargs):
        # 从 MRO 中提取原始类
        orig_cls = cls.__mro__[_orig_cls_mro_index]  # 默认 index=2，即 SimpleMLP
        instance = object.__new__(orig_cls)
        if _enable_fsdp_module_new_init:
            orig_cls.__init__(instance, *args, **kwargs)
        return instance
```

### 2.4 参数组初始化

```python
# _fully_shard.py: _init_param_group()
def _init_param_group(state, params, modules, mesh_info, ...):
    state._fsdp_param_groups.append(
        FSDPParamGroup(
            params, modules, mesh_info,
            post_forward_mesh_info, device,
            shard_placement_fn, mp_policy, offload_policy,
        )
    )
```

每个 `FSDPParamGroup` 管理一组参数的分片/非分片生命周期。

### 2.5 参数即刻分片（Lazy Init）

```python
# _fsdp_param_group.py: FSDPParamGroup.lazy_init()
# 在首次 _pre_forward 时调用
def lazy_init(self):
    for fsdp_param in self.fsdp_params:
        fsdp_param.init_sharded_param()  # 创建 DTensor 分片参数
    self._sharded_state = ShardedState.SHARDED
```

**init_sharded_param 内部**：

```python
# _fsdp_param.py: FSDPParam._init_sharded_param()
def _init_sharded_param(self):
    # 1. 确定分片维度（默认 dim=0）
    fsdp_placement = self.fsdp_placement or Shard(0)
    
    # 2. 对原始参数沿 dim=0 分片
    # param shape: (1024, 512) -> 分片后每 rank 得到 (256, 512)（4 GPU 时）
    padded_shard = param_data.new_zeros(padded_sharded_size)
    param_data_chunk = param_data.chunk(world_size, dim=shard_dim)[rank]
    padded_shard.narrow(...).copy_(param_data_chunk)
    
    # 3. 展平为 1D（便于 all-gather 通信）
    self._sharded_param_data = padded_shard.flatten()
    
    # 4. 包装为 DTensor
    self._sharded_param = to_sharded_dtensor(...)
```

### 2.6 状态上下文初始化

```python
# 每个 FSDPState 维护一个 FSDPStateContext
class FSDPStateContext:
    is_last_backward: bool = True   # 微批次支持
    reduce_grads: bool = True       # 是否执行 reduce-scatter
    all_reduce_grads: bool = True   # 是否执行 all-reduce（HSDP）
```

## 3. `fully_shard(model)` vs `fully_shard(layer)` 的区别

| 特性 | `fully_shard(model)` | `fully_shard(layer)` |
| ------ | --------------------- | --------------------- |
| 参数粒度 | 所有参数在一个分片组 | 每层参数独立分片组 |
| 通信次数 | 一次 all-gather → 一次 reduce-scatter | 每层各自通信 |
| 内存峰值 | 一次性 all-gather 所有参数，内存高 | 逐层 all-gather，内存低 |
| 通信-计算重叠 | 差（需要等所有参数） | 好（可逐层重叠） |
| 前向预取 | 无（只有一个模块） | 有（下一层的 all-gather 与当前层计算重叠） |

**最佳实践**：对 Transformer 的每个 Block 分别 `fully_shard`，对 embedding 和 lm_head 也分别包装。顶层 `model` 也需要 `fully_shard` 以设置根状态。

## 4. 参数在分片前后的状态

```python
# 分片前
fc1.weight: torch.Size([512, 1024])  # 完整参数，每个 GPU 都有

# 分片后（4 GPU, Shard(0)）
fc1.weight: DTensor(
    local_tensor: torch.Size([128, 1024]),  # 每个 GPU 只有 1/4
    placements: (Shard(0),),
    mesh: (4,)
)

# 前向计算时（all-gather 后）
fc1.weight: torch.Size([512, 1024])  # 临时恢复完整参数

# 前向计算后（reshard）
fc1.weight: DTensor(
    local_tensor: torch.Size([128, 1024]),  # 恢复分片
    placements: (Shard(0),),
)
```

## 5. fully_shard 的关键参数说明

```python
fully_shard(
    module,
    mesh=None,                    # DeviceMesh，1D=FSDP, 2D=HSDP
    reshard_after_forward=True,   # 前向后是否释放完整参数
    shard_placement_fn=None,      # 自定义分片维度函数
    mp_policy=MixedPrecisionPolicy(),  # 混合精度策略
    offload_policy=OffloadPolicy(),    # CPU offload 策略
    ignored_params=None,          # 不分片的参数（如可训练 norm）
)
```

### reshard_after_forward 详解

- `True`：前向计算后立即释放完整参数，节省内存（默认）
- `False`：保持参数完整，后续前向无需再 all-gather
- `int`：部分 reshard，例如 `2` 表示在 2 个 GPU 间复制参数

### 混合精度策略

```python
from torch.distributed.fsdp import MixedPrecisionPolicy

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,  # all-gather 时的参数精度
    reduce_dtype=torch.float32,  # reduce-scatter 时的梯度精度
)
```

## 6. 完整训练循环示例

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

# 初始化
dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)

mesh = init_device_mesh("cuda", (dist.get_world_size(),))

# 模型
class TransformerBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10000, 512)
        self.blocks = nn.Sequential(*[TransformerBlock(512, 2048) for _ in range(6)])
        self.lm_head = nn.Linear(512, 10000)

    def forward(self, x):
        x = self.embed(x)
        x = self.blocks(x)
        return self.lm_head(x)

model = Model().cuda()

# 逐层 FSDP 包装（最佳实践）
for block in model.blocks:
    fully_shard(block, mesh=mesh)
fully_shard(model.embed, mesh=mesh)
fully_shard(model.lm_head, mesh=mesh)
# 顶层包装 — 设置根状态，管理 backward 最终回调
fully_shard(model, mesh=mesh)

# 训练
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(3):
    x = torch.randint(0, 10000, (32, 128)).cuda()  # (batch, seq_len)
    y = torch.randint(0, 10000, (32, 128)).cuda()

    optimizer.zero_grad()
    output = model(x)
    loss = nn.functional.cross_entropy(
        output.view(-1, 10000), y.view(-1)
    )
    loss.backward()
    optimizer.step()
    # 注意: FSDP2 在 optimizer.step() 之后自动 reshared 参数
```

## 7. 关键 API 速查

| API | 作用 |
| ----- | ------ |
| `fully_shard(module, mesh)` | 对模块应用 FSDP |
| `fully_shard.state(module)` | 获取模块的 FSDPState |
| `module.reshard()` | 手动释放完整参数 |
| `module.unshard()` | 手动 all-gather 参数 |
| `module.set_is_last_backward(False)` | 微批次支持：非最后一个微批次 |
| `module.set_requires_gradient_sync(False)` | 等价 FSDP1 的 `no_sync` |
| `module.set_modules_to_forward_prefetch([...])` | 自定义前向预取目标 |
| `module.set_modules_to_backward_prefetch([...])` | 自定义反向预取目标 |

---

> **源码参考**：
>
> - `fully_shard` 入口：`torch/distributed/fsdp/_fully_shard/_fully_shard.py`
> - `FSDPModule` 类：同上文件
> - MRO 替换机制：`_apply_to_module()` in `_fully_shard.py`
> - 参数组初始化：`_init_param_group()` in `_fsdp_init.py`
> - FSDPParam 分片：`_fsdp_param.py`
