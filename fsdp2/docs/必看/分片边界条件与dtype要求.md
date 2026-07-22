# 分片边界条件与 dtype 要求：dim 0 不可切 / 管理范围 / dtype 限制

> 基于 PyTorch v2.12.0 源码分析
> 核心源码：
>
> - `_fsdp_param.py`：`_init_sharded_param` / `init_dtype_attrs` / `_verify_managed_param`
> - `_fsdp_common.py`：`_chunk_with_empty` / `_get_dim0_padded_size` / `_get_dim_chunked_size`
> - `_fsdp_init.py`：`_get_managed_states` / `_move_states_to_device` / `_verify_managed_param`
> - `_fsdp_collectives.py`：`foreach_all_gather` / `foreach_reduce_scatter` / `_get_dim0_padded_size`
> - `_fully_shard.py`：`fully_shard` docstring

## 0. 问题汇总

本文件回答 5 个核心问题：

1. 当 `fully_shard` 的 dim 0 不够切时（如 tensor `[8,128,1280]`，world_size=16）会发生什么？
2. 针对这种情况，最佳实践是什么？PyTorch 官方有指导吗？
3. `fully_shard` 会管理哪些内容？（nn.Module / nn.Parameter / ...）
4. 对 tensor 有 dtype 要求吗？
5. `fully_shard` 不会切分哪些东西？

## 1. dim 0 不可切分时的行为

### 1.1 示例场景

```python
import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

# 假设 16 GPU
mesh = init_device_mesh("cuda", (16,))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # dim 0 = 8 < world_size 16，无法整除
        self.weight = nn.Parameter(torch.randn(8, 128, 1280))

model = Model().cuda()
fully_shard(model, mesh=mesh)  # 会发生什么？
```

### 1.2 结论：不会报错，自动 padding

**核心机制**：dim 0 的 uneven sharding 是**显式允许**的，通过 padding 处理；只有非零维度的 uneven sharding 才会报错。

#### 证据 A：源码中对 dim 0 和非零维的差异化处理

`_fsdp_param.py` 第 285-293 行：

```python
# _fsdp_param.py: FSDPParam._init_sharded_param()
if shard_dim > 0 and param_data.size(shard_dim) % shard_world_size != 0:
    # If sharding on nonzero dim, require even sharding for now because
    # the uneven sharding (1) requires extra copies before/after FSDP
    # collectives and (2) introduces extra complexity to handle padding
    # and unpadding
    raise NotImplementedError(
        f"FSDP does not support uneven sharding on dim {shard_dim}: "
        f"{param_data.size()} (world size: {shard_world_size})"
    )
chunks = _chunk_with_empty(param_data, shard_world_size, dim=shard_dim)
```

**关键点**：条件是 `shard_dim > 0`，即只有非零维度要求整除。dim 0 的 uneven sharding 被显式允许。注释解释了原因：非零维的 uneven sharding 需要额外的 copy 和复杂的 padding/unpadding 逻辑；而 dim 0 的 uneven sharding 可以通过简单的 pre-padding 处理。

#### 证据 B：`_chunk_with_empty` 用空 tensor 填充

`_fsdp_common.py` 第 115-121 行：

```python
def _chunk_with_empty(
    tensor: torch.Tensor, num_chunks: int, dim: int
) -> list[torch.Tensor]:
    chunks = list(torch.chunk(tensor, num_chunks, dim=dim))
    while len(chunks) < num_chunks:
        chunks.append(chunks[0].new_empty(0))
    return chunks
```

对 `[8, 128, 1280]` 调用 `torch.chunk(..., 16, dim=0)` 只会返回 8 个 `[1, 128, 1280]` 的 chunk，然后用 `new_empty(0)` 补到 16 个。所以 **rank 0-7 拿到 `[1, 128, 1280]`，rank 8-15 拿到 `[0, 128, 1280]`（空）**。

#### 证据 C：pre-padding 到均匀大小

`_fsdp_param.py` 第 299-311 行：

```python
self.contiguous_sharded_stride = make_contiguous_strides_for(self.sharded_size)
padded_sharded_size = chunks[0].size()  # 0th always padded
self.padded_sharded_param_size = padded_sharded_size
# Pre-pad the sharded parameter to avoid padding before all-gather
padded_sharded_param = param_data.new_zeros(padded_sharded_size)
if sharded_param.numel() > 0:
    padded_sharded_param.narrow(
        dim=shard_dim, start=0, length=sharded_param.size(shard_dim)
    ).copy_(sharded_param)
```

每个 rank 的 sharded param 被 pre-pad 到 `chunks[0].size()`（即 `[1, 128, 1280]`）。这样 all-gather 前就不需要再 pad，避免运行时开销。

#### 证据 D：all-gather 后用 `as_strided` 去 padding

`_fsdp_param.py` 第 608-613 行：

```python
unsharded_param = torch.as_strided(
    unsharded_tensor,           # [16, 128, 1280] all-gather 输出（含 padding 零）
    self._orig_size,           # [8, 128, 1280] 原始大小
    self._contiguous_orig_stride,
    storage_offset=0,
)
```

all-gather 后得到 `[16, 128, 1280]`（含 padding 的零），用 `as_strided` 还原成原始 shape `[8, 128, 1280]`。**注意**：这要求底层 storage 连续且 `as_strided` 的 view 不会越界——因为 `_orig_size` 的 numel（8×128×1280）小于等于 all-gather 输出的 numel（16×128×1280）。

### 1.3 具体流程示例

以 `[8, 128, 1280]`，world_size=16 为例：

| 阶段 | rank 0-7 | rank 8-15 |
| ------ | ---------- | ----------- |
| `torch.chunk(dim=0, 16)` | `[1, 128, 1280]` 真实数据 | 不存在（只有 8 个 chunk） |
| `_chunk_with_empty` 补齐 | `[1, 128, 1280]` | `[0, 128, 1280]`（空 tensor） |
| pre-padding | 已是 `[1, 128, 1280]`，无需 pad | pre-pad 到 `[1, 128, 1280]`（全零） |
| `_sharded_param_data` | 真实数据 1×128×1280 | 全零 1×128×1280 |
| all-gather | 贡献真实数据 | 贡献零数据 |
| `as_strided` 还原 | → `[8, 128, 1280]` | → `[8, 128, 1280]` |

### 1.4 代价分析

虽然能正常工作，但有显著代价：

1. **通信浪费**：rank 8-15 白白参与 all-gather，贡献的全是零，浪费 50% 的通信带宽。
2. **内存浪费**：rank 8-15 持有全零的 sharded param，没有实际数据但占内存。
3. **负载不均**：部分 rank 无有效数据，参数更新时也是对零 tensor 操作。

### 1.5 官方测试验证

PyTorch 测试用例明确验证了 dim 0 远小于 world_size 的场景。`test_fully_shard_init.py` 第 678-679 行：

```python
# Test both even sharding (8), uneven sharding (3), and empty local tensor (1)
for mlp_dim in (8, 3, 1):
    # cover foreach_copy code path for bf16
    for mp_policy in (
        MixedPrecisionPolicy(),
        MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        ),
    ):
        with torch.device("meta"):
            model = nn.Sequential(
                MLP(mlp_dim, dim_multiplier=1, with_buffer=True, bias=False),
                MLP(mlp_dim, dim_multiplier=1, bias=False),
            )
            ...
            fully_shard(model[0], mesh=mesh, mp_policy=mp_policy)
            fully_shard(model[1], mesh=mesh, mp_policy=mp_policy)
            fully_shard(model, mesh=mesh, mp_policy=mp_policy)
```

`mlp_dim=1` 的场景下，所有参数 dim 0 = 1，在多 GPU 下大部分 rank 持有空 tensor。这被作为正常测试场景，说明 **dim 0 的 uneven sharding 是被设计支持的行为，而非 edge case**。

同时，`test_shard_tensor_parameters` 测试（第 424-429 行）用 `MLP(3, dim_multiplier=3)` 测试 odd dim sizes：

```python
def test_shard_tensor_parameters(self):
    # Use odd dim sizes to test uneven shards
    model = nn.Sequential(*[MLP(3, dim_multiplier=3) for _ in range(3)])
    orig_params = [param.detach().clone() for param in model.parameters()]
    fully_shard(model)
    sharded_params = list(model.parameters())
    self._check_1d_sharded_parameters(orig_params, sharded_params)
```

---

## 2. 最佳实践与官方指导

### 2.1 官方 docstring 指导

`fully_shard` 的 docstring（`_fully_shard.py` 第 131-144 行）明确说明了分片规则：

> ```python
> shard_placement_fn (Optional[Callable[[nn.Parameter], Optional[Shard | ShardPlacementResult]]]):
>     ...
>     If sharding on a nonzero dim, we currently require even sharding,
>     i.e. the tensor dim size on that dim must be divisible by the FSDP
>     shard mesh size.
> ```

即：**非零维度必须整除，dim 0 可以 uneven（自动 padding）**。

### 2.2 推荐做法（按优先级）

#### 做法 1：用 `shard_placement_fn` 指定一个能整除的维度（推荐）

如果参数有某个维度能被 world_size 整除，优先切那个维度。

```python
from torch.distributed.tensor import Shard

# [8, 128, 1280] in 16 GPU
# dim 0: 8 % 16 != 0  ✗
# dim 1: 128 % 16 == 0 ✓
# dim 2: 1280 % 16 == 0 ✓

def shard_fn(param):
    # 对 [8, 128, 1280] 的参数，切 dim=1
    return Shard(1)

fully_shard(module, shard_placement_fn=shard_fn)
```

**注意**：非零维度**必须整除**，否则会触发 `_fsdp_param.py` 第 290 行的 `NotImplementedError`。源码注释说明了原因：

> uneven sharding (1) requires extra copies before/after FSDP collectives and (2) introduces extra complexity to handle padding and unpadding

#### 做法 2：接受 dim 0 padding（当所有维度都不能整除时）

如果所有维度都不能被 world_size 整除，dim 0 是唯一能 uneven shard 的维度，直接用默认的 `Shard(0)` 即可，FSDP 会自动 padding。

```python
# [7, 100, 50] in 16 GPU
# 所有维度都不能整除，只能切 dim 0
fully_shard(module)  # 默认 Shard(0)，自动 padding
```

#### 做法 3：用 HSDP（2D mesh）减小 shard group

如果 world_size 远大于 dim 0，考虑用 HSDP，让部分 rank 做 replicate 而非 shard：

```python
from torch.distributed.device_mesh import init_device_mesh

# 16 GPU，dim 0 = 8，shard group 只用 8 个 rank
mesh = init_device_mesh("cuda", (2, 8), mesh_dim_names=("replicate", "shard"))
fully_shard(model, mesh=mesh)
# shard group = 8，刚好整除 dim 0 = 8
```

#### 做法 4：用 `reshard_after_forward` int 形式减小 shard group

`reshard_after_forward` 支持传入 int，表示 reshard 到更小的 world size（`_fsdp_init.py` 第 112-140 行）：

```python
# 16 GPU，但 forward 后只 reshard 到 8 个 rank
fully_shard(module, reshard_after_forward=8)
```

但这只影响 forward 后的 reshard，不影响初始化时的 sharding。

### 2.3 实际场景建议

| 场景 | 推荐做法 |
| ------ | --------- |
| 参数 dim 0 < world_size，但其他维度能整除 | `shard_placement_fn=Shard(N)` 切其他维度 |
| 参数所有维度都不能整除 | 默认 `Shard(0)`，接受 padding 浪费 |
| world_size 远大于 dim 0 | HSDP 减小 shard group |
| 小参数（如 bias） | 考虑 `ignored_params` 跳过分片 |

---

## 3. `fully_shard` 会管理哪些内容？

### 3.1 管理范围总览

| 对象 | 是否管理 | 管理方式 |
| ------ | --------- | --------- |
| `nn.Module` | 是 | 动态修改类 MRO，注入 `FSDPModule` |
| `nn.Parameter` | 是 | 切分成 DTensor，注册回 module |
| `torch.Tensor` buffer | 部分 | 仅 move 到 device，**不参与分片** |
| 梯度 | 是 | backward 时 reduce-scatter |
| Optimizer states | 是 | 通过优化器钩子管理 |
| `ignored_params` | 否 | 显式排除 |

### 3.2 nn.Module 管理

`_fsdp_init.py` 第 504-518 行，`_apply_to_module` 动态修改 module 的类：

```python
def _apply_to_module(
    modules: tuple[nn.Module, ...],
    cls_to_wrapper_cls: dict[type, type],
    wrapper_module_cls: type,        # FSDPModule
    wrapper_cls_prefix: str,         # "FSDP"
    unimplemented_deepcopy: "Callable",
) -> None:
    for module in modules:
        cls = module.__class__
        new_cls = cls_to_wrapper_cls.get(cls)
        if not new_cls:
            dct = {"__deepcopy__": unimplemented_deepcopy}
            new_cls = type(
                f"{wrapper_cls_prefix}{cls.__name__}", (wrapper_module_cls, cls), dct
            )
            cls_to_wrapper_cls[cls] = new_cls
        module.__class__ = new_cls
```

即原 `MyModule` 的 `__class__` 被改成 `FSDPMyModule(FSDPModule, MyModule)`，通过 MRO 让 `FSDPModule` 的方法优先级最高。

### 3.3 nn.Parameter 管理

`_fsdp_param.py` 第 319-322 行，切分后的 DTensor 参数被写回 module：

```python
self.sharded_param = nn.Parameter(
    self.to_sharded_dtensor(sharded_param),
    requires_grad=param.requires_grad,
)
# Let `param_data` be freed normally when its ref count reaches 0 when
# the `fully_shard` call returns to allow provided parameters to alias
self._setattr_on_modules(self.sharded_param)
self.sharded_state = ShardedState.SHARDED
```

`_setattr_on_modules` 会同时更新原 module 和共享该参数的其他 module（`_fsdp_param.py` 第 670-677 行）：

```python
def _setattr_on_modules(self, param: nn.Parameter) -> None:
    unsafe_setattr_param(
        self._module_info.module, self._module_info.param_name, param
    )
    for shared_module, shared_param_name in zip(
        self._module_info.shared_modules, self._module_info.shared_param_names
    ):
        unsafe_setattr_param(shared_module, shared_param_name, param)
```

### 3.4 Buffer 管理（特殊：只 move，不 shard）

`_fsdp_init.py` 第 338-366 行，`_get_managed_states` 收集 params 和 buffers：

```python
def _get_managed_states(
    modules: list[nn.Module], ignored_params: set[nn.Parameter] | None = None
) -> tuple[list[nn.Parameter], list[torch.Tensor]]:
    params: list[nn.Parameter] = []
    buffers: list[torch.Tensor] = []
    visited_params: set[nn.Parameter] = set()
    visited_buffers: set[torch.Tensor] = set()
    if ignored_params is None:
        ignored_params = set()

    for module in modules:
        for name, param in module.named_parameters(recurse=False):
            if param in ignored_params:
                continue
            if param not in visited_params:
                _verify_managed_param(name, param)
                params.append(param)
                visited_params.add(param)
        for buffer in module.buffers(recurse=False):  # ← 收集 buffer
            if buffer not in visited_buffers:
                buffers.append(buffer)
                visited_buffers.add(buffer)
    return params, buffers
```

`_fsdp_init.py` 第 368-389 行，`_move_states_to_device` 只做 `.to(device)`，**不调用任何 sharding 逻辑**：

```python
def _move_states_to_device(
    params: list[nn.Parameter],
    buffers: list[torch.Tensor],
    device: torch.device,
) -> None:
    for tensor in itertools.chain(params, buffers):
        if tensor.device == device or tensor.device.type == "meta":
            continue
        if isinstance(tensor, DTensor):
            ...
        tensor_ = tensor
        if is_traceable_wrapper_subclass(tensor_):
            with torch.no_grad():
                tensor_on_device = nn.Parameter(tensor.to(device))
            torch.utils.swap_tensors(tensor, tensor_on_device)
        else:
            tensor.data = tensor.to(device)  # ← 只移动，不分片
```

**关键证据**：buffers 只出现在 `_move_states_to_device` 中，从未传入 `_init_param_group`（分片逻辑）。这意味着 **buffer 不参与 all-gather / reduce-scatter，每个 rank 持有完整的 buffer 副本**。

### 3.5 有 buffer 的 module 不能被 ignore

`_fsdp_init.py` 第 235-242 行：

```python
def _ignore_module(
    module: nn.Module,
    ignored_params: set[nn.Parameter],
    ignore_decision: dict[nn.Module, bool],
) -> bool:
    if module in ignore_decision:
        return ignore_decision[module]

    if len(list(module.buffers(recurse=False))) > 0:
        # Cannot ignore a module with any buffer
        ignore_decision[module] = False
        return False
    ...
```

即：**有 buffer 的 module 不能被 ignore**，因为 FSDP 需要管理（move）这些 buffer。这进一步证明 buffer 被 FSDP 管理，只是管理方式与 param 不同。

### 3.6 梯度管理

backward 时，FSDP 通过 hook 调用 `foreach_reduce_scatter`（`_fsdp_collectives.py` 第 570+ 行），把 unsharded 梯度 reduce-scatter 成 sharded 梯度：

```python
# _fsdp_collectives.py: foreach_reduce_scatter
padded_unsharded_sizes = tuple(
    _get_dim0_padded_size(grad.size(), world_size) for grad in unsharded_grads
)
reduce_scatter_input_numel = sum(s.numel() for s in padded_unsharded_sizes)
reduce_scatter_output_numel = reduce_scatter_input_numel // world_size
...
```

### 3.7 Optimizer states 管理

通过 `_fsdp_optim_utils.py` 和 FSDP 的 optimizer hook 管理，确保 optimizer states（如 Adam 的 momentum）也被切分。这部分不在本文件讨论范围。

---

## 4. 对 tensor 的 dtype 要求

### 4.1 结论：无硬性 dtype 限制

`fully_shard` 对参数 dtype **没有硬性限制**，支持 fp32 / bf16 / fp16 / fp8 / int 等各种 dtype。唯一的限制是 scalar 参数会被拒绝。

### 4.2 唯一的 dtype 相关校验：mixed precision 只对浮点生效

`_fsdp_param.py` 第 527-541 行，`init_dtype_attrs`：

```python
def init_dtype_attrs(self, mp_policy: MixedPrecisionPolicy):
    param_dtype, reduce_dtype = (mp_policy.param_dtype, mp_policy.reduce_dtype)
    self.orig_dtype = self.sharded_param.dtype
    # Clamp `reduce_dtype` to `None` if no casting is required
    if reduce_dtype == param_dtype:
        reduce_dtype = None
    # Clamp `param_dtype` to `None` if no casting is required or if the
    # parameter is non-floating-point (mixed precision is only meaningful
    # for floating-point parameters)
    if param_dtype == self.orig_dtype or not self.orig_dtype.is_floating_point:
        param_dtype = None
    self.param_dtype = param_dtype
    self.reduce_dtype = reduce_dtype
    # None indicates that the mixed precision is not enabled
```

**关键点**：`not self.orig_dtype.is_floating_point` 时，`param_dtype` 被设为 `None`，即非浮点参数（int / bool 等）不参与混合精度转换，保留原 dtype。

### 4.3 支持多种 dtype 的证据

#### 证据 A：bf16 / fp32 混合精度是常见场景

`_fsdp_collectives.py` 第 386 行注释：

```python
# Intentionally try to run a fast-path that bypasses abstractions for the
# common FSDP case of bf16/fp32 mixed precision in order to use foreach
# copy for lower CPU overhead and more efficient copying in eager
```

#### 证据 B：uint8 专门处理（fp8 量化场景）

`_fsdp_collectives.py` 第 352-356 行：

```python
if dtype == torch.uint8:
    all_gather_inputs = [
        t.view(torch.uint8) for ts in param_all_gather_inputs for t in ts
    ]
else:
    all_gather_inputs = [*chain.from_iterable(param_all_gather_inputs)]
```

`_fsdp_collectives.py` 第 801-803 行：

```python
# For fp32/bf16, we do not need to worry about overflow/underflow, so we
# use NCCL's built-in division to avoid separate div kernels
overflow_risk = reduce_dtype not in (torch.float32, torch.bfloat16)
```

#### 证据 C：不同参数可以有不同 dtype

FSDP2 的 DTensor 表示允许同一个模型中混合不同 dtype 的参数。HuggingFace 文档也提到：

> FSDP2 supports mixing fp8 and other parameter types in the same model out of the box

### 4.4 唯一被拒绝的：scalar 参数

`_fsdp_init.py` 第 325-332 行，`_verify_managed_param`：

```python
def _verify_managed_param(name: str, param: nn.Parameter) -> None:
    """
    Verify if the parameter is accepted by fully_shard. The only restriction now
    is that the parameter cannot be a scalar tensor (param.numel == 0) since we
    need at least one dim to shard.
    """
    if len(param.shape) == 0:
        raise ValueError(
            "fully_shard doesn't support scalar parameters. "
            f"Change {name} to a 1D tensor with numel equal to 1."
        )
```

注意：docstring 说 `param.numel == 0` 但实际检查的是 `len(param.shape) == 0`（即 0 维 tensor）。修复方式是改成 1D tensor（即使 numel=1 也可以）。

### 4.5 非连续参数也被拒绝

`_fsdp_param.py` 第 262-264 行：

```python
if not param.is_contiguous():
    raise NotImplementedError(
        f"FSDP does not support non-contiguous parameters yet: {param.shape=} {param.stride()=}"
    )
```

测试用例 `test_raise_noncontiguous_parameter`（`test_fully_shard_init.py` 第 453-460 行）验证：

```python
def test_raise_noncontiguous_parameter(self):
    conv2d = nn.Conv2d(8, 8, 3).to(memory_format=torch.channels_last)
    with self.assertRaisesRegex(
        NotImplementedError, "FSDP does not support non-contiguous parameters"
    ):
        fully_shard(conv2d)
```

### 4.6 dtype 推荐配置

| 场景 | 推荐配置 |
| ------ | --------- |
| 标准 bf16 训练 | `mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)` |
| 纯 fp32 训练 | `mp_policy=MixedPrecisionPolicy()`（默认） |
| fp8 量化训练 | 参数本身是 fp8 dtype，FSDP 原生支持 |
| 非 float 参数（如 int） | 不需要特殊配置，FSDP 保留原 dtype |

---

## 5. `fully_shard` 不会切分哪些东西？

### 5.1 不会被切分的内容总览

| 内容 | 原因 | 证据位置 |
| ------ | ------ | --------- |
| **Buffers** | 只 move 到 device，不传入分片逻辑 | `_fsdp_init.py` 第 362-364, 371-389 行 |
| **`ignored_params` 中的参数** | 显式 `continue` 跳过 | `_fsdp_init.py` 第 353-354 行 |
| **已被先前 `fully_shard` 分组的参数** | DFS 跳过已有 FSDP state 的子 module | `_fsdp_init.py` 第 262-268 行 |
| **Scalar 参数** | 直接 raise `ValueError` | `_fsdp_init.py` 第 325-332 行 |
| **非连续参数** | 直接 raise `NotImplementedError` | `_fsdp_param.py` 第 262-264 行 |
| **`nn.ModuleList` / `nn.ModuleDict` 等无 forward 的容器** | 直接 raise `ValueError` | `_fsdp_init.py` 第 30-37 行 |
| **DTensor 参数（已分片）** | 走特殊路径复用已有 placement | `_fsdp_param.py` `_init_sharding_spec_tp` |

### 5.2 证据详解

#### 5.2.1 Buffers 不被切分

`_fsdp_init.py` 第 565-567 行，`_get_modules_and_states` 返回 params 和 buffers，但只有 params 传入 `_init_param_group`：

```python
def _get_modules_and_states(...) -> ...:
    ...
    params, buffers = _get_managed_states(managed_modules, ignored_params)
    _move_states_to_device(params, buffers, device)  # ← buffers 只 move
    return arg_module, modules, managed_modules, params, buffers
```

`_fully_shard.py` 第 247-258 行，`fully_shard` 函数中只有 `params` 传入 `_init_param_group`：

```python
arg_module, modules, managed_modules, params, buffers = _get_modules_and_states(
    module, device, ignored_params
)
state = fully_shard.state(modules[0])
state.init(modules, device, mp_policy, auto_reshard_after_forward)

_init_param_group(   # ← 只有 params
    state,
    params,          # ← 没有 buffers
    modules,
    ...
)
```

**结论**：buffers 只 move 到 device，每个 rank 持有完整副本，不参与 all-gather / reduce-scatter。

#### 5.2.2 `ignored_params` 被显式跳过

`_fsdp_init.py` 第 353-354 行：

```python
for name, param in module.named_parameters(recurse=False):
    if param in ignored_params:
        # do not include an ignored parameters
        continue
    ...
```

被 ignore 的参数：

- 不被切分
- 不被 move 到 device（因为根本不在 params 列表中）
- 不参与梯度 reduce-scatter

#### 5.2.3 已被先前 `fully_shard` 分组的参数

`_fsdp_init.py` 第 262-268 行，`_get_managed_modules` 的 DFS：

```python
def dfs(module: nn.Module) -> None:
    if not is_composable_fn(module):
        return
    elif module not in root_modules_set and get_state_fn(module) is not None:
        return  # nested `fully_shard` module  ← 跳过已有 FSDP state 的子 module
    visited_modules.add(module)
    for submodule in module.children():
        if submodule not in visited_modules:
            dfs(submodule)
    modules.append(module)
```

这就是为什么 `fully_shard` 要 **bottom-up 调用**：先 wrap 子 module，子 module 的参数就被分组了；再 wrap 父 module 时，父 module 只能拿到尚未分组的参数。

#### 5.2.4 Scalar 参数被拒绝

`_fsdp_init.py` 第 325-332 行（详见 4.4 节）。

#### 5.2.5 非连续参数被拒绝

`_fsdp_param.py` 第 262-264 行（详见 4.5 节）。

#### 5.2.6 无 forward 的容器被拒绝

`_fsdp_init.py` 第 30-37 行：

```python
def _validate_module(module: nn.Module, func_name: str) -> None:
    """
    Validate that the module can be used with fully_shard or replicate.

    Raises ValueError if the module is a container that doesn't implement forward.
    """
    if (
        isinstance(module, (nn.ModuleList, nn.ModuleDict))
        and module.__class__.forward is nn.Module.forward
    ):
        raise ValueError(
            f"{func_name} does not support containers that do not implement forward: {module}"
        )
```

即 `nn.ModuleList` / `nn.ModuleDict` 这类纯容器（没有自定义 forward）不能直接 `fully_shard`。

#### 5.2.7 DTensor 参数走特殊路径

如果参数已经是 DTensor（如 TP/EP 场景），`_init_sharded_param` 会调用 `_init_sharding_spec_tp` 或 `_init_sharding_spec_spmd`（`_fsdp_param.py` 第 466-549 行），复用已有的 placement，而非重新切 dim 0。

```python
def _init_sharding_spec(self, param, fsdp_placement, shard_dim) -> torch.Tensor:
    self._unsharded_dtensor_spec = None
    if self.mesh_info.is_spmd_mesh and not self.is_dtensor:
        raise ValueError(...)  # SPMD mesh 下必须是 DTensor
    if self.is_dtensor and self.mesh_info.is_spmd_mesh:
        return self._init_sharding_spec_spmd(param, fsdp_placement, shard_dim)
    if self.is_dtensor:
        return self._init_sharding_spec_tp(param, fsdp_placement, shard_dim)
    return self._init_sharding_spec_plain(param, fsdp_placement)  # 普通张量路径
```

---

## 6. 总结速查表

| 问题 | 答案 |
| ------ | ------ |
| dim 0 < world_size 会报错吗？ | 不会，自动 padding（rank 多的拿空 tensor） |
| 非零维不能整除会报错吗？ | 会，`NotImplementedError` |
| 最佳实践？ | 优先用 `shard_placement_fn` 切能整除的维度；或用 HSDP 减小 shard group |
| 管理哪些对象？ | nn.Module（注入 FSDPModule）、nn.Parameter（切分成 DTensor）、buffer（只 move）、梯度、optimizer states |
| dtype 限制？ | 无限制，但 mixed precision 只对浮点生效；scalar 参数被拒绝 |
| 不切分哪些？ | buffer、ignored_params、已被分组的参数、scalar、非连续参数、无 forward 的容器 |
