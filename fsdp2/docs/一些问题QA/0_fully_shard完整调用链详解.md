# fully_shard 完整调用链详解：从 API 调用到前向计算

> 基于 PyTorch v2.12.0 源码分析
> 核心源码入口：[`_fully_shard.py`](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/distributed/fsdp/_fully_shard/_fully_shard.py)

## 0. 贯穿全文的示例代码

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)

mesh = init_device_mesh("cuda", (4,))  # 4 GPU FSDP

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = Block()
        self.block1 = Block()

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        return x

model = Model().cuda()
fully_shard(model.block0, mesh=mesh)  # ① FSDP 包装 block0
fully_shard(model.block1, mesh=mesh)  # ② FSDP 包装 block1
fully_shard(model, mesh=mesh)         # ③ 顶层包装

x = torch.randn(32, 512).cuda()
output = model(x)  # ④ 完整前向计算
```

本文按①②③④四个关键时间点，逐步拆解每一步发生的事件。

---

## 1. 前置知识：fully_shard 的核心设计思想

在深入源码之前，先理解 FSDP2 的四个核心设计：

### 1.1 MRO 替换（类层次结构注入）

FSDP2 **不依赖 PyTorch 的 forward hook**（与 FSDP1 的关键区别），而是直接**修改模块的 `__class__`**，将 `FSDPModule` 插入到模块的方法解析顺序（MRO）中。

```text
修改前: Block -> nn.Module -> ...
修改后: FSDPBlock -> FSDPModule -> Block -> nn.Module -> ...
```

`FSDPModule` 在 MRO 中位于原始类之前，因此 `FSDPModule.forward`（如果定义了）会先于 `Block.forward` 被调用。但 FSDP2 并不重写 `forward`，而是通过**在 `_pre_forward` / `_post_forward` 中包装整个调用**来实现拦截。

### 1.2 @contract 装饰器

```python
@contract(state_cls=FSDPState)
def fully_shard(module, ...):
```

`@contract` 装饰器为传入的 module 附加一个 `FSDPState` 对象，可通过 `fully_shard.state(module)` 访问。**state 和 module 是 1:1 的关系**。

### 1.3 参数组（FSDPParamGroup）

`fully_shard` 将模块的所有参数打包成一个 `FSDPParamGroup`，该组内的所有参数共享一次 all-gather 和一次 reduce-scatter 通信。这是 FSDP2 将"通信分组"提升为一等公民的设计。

### 1.4 自底向上（Bottom-up）调用顺序

`fully_shard` 必须先应用于子模块，再应用于父模块。父模块的参数组只包含那些尚未被子模块 `fully_shard` 管理的参数。

---

## 2. 第一次调用：`fully_shard(model.block0, mesh=mesh)`

### 2.1 入口函数流程

```python
# _fully_shard.py: fully_shard()
@contract(state_cls=FSDPState)
def fully_shard(module, *, mesh=None, reshard_after_forward=None, ...):
    # Step 1: 日志记录
    torch._C._log_api_usage_once("torch.distributed.fsdp.fully_shard")

    # Step 2: 模块验证
    _validate_module(module, "fully_shard")
```

#### Step 2 详解：`_validate_module`

```python
# _fsdp_init.py
def _validate_module(module: nn.Module, func_name: str) -> None:
    # 检查是否被多次调用 fully_shard（幂等性保护）
    if isinstance(module, FSDPModule):
        raise ValueError(
            f"FSDP module cannot be passed to {func_name} again"
        )
    # 检查是否为不支持的容器类型
    if isinstance(module, (nn.ModuleList, nn.ModuleDict, nn.ParameterList,
                           nn.ParameterDict)):
        raise ValueError(
            f"{func_name} does not support {type(module)}"
        )
```

对于 `model.block0`（一个 `Block` 实例），验证通过。

```python
    # Step 3: Mesh 处理
    mesh = mesh or _init_default_mesh()
    _validate_mesh(mesh, dp_mesh_dims)
    mesh_info = _get_mesh_info(mesh, dp_mesh_dims)
```

#### Step 3 详解：Mesh 信息解析

```python
# _fsdp_init.py
def _get_mesh_info(mesh, dp_mesh_dims=None):
    # mesh 是 1D DeviceMesh("cuda", (4,))
    # mesh.ndim == 1, dp_mesh_dims is None
    # → 返回 FSDPMeshInfo
    return FSDPMeshInfo(
        mesh=mesh,                # DeviceMesh("cuda", (4,))
        shard_mesh_dim=0,         # 分片维
        shard_mesh_size=4,        # 分片组大小 = 4
        shard_process_group=mesh.get_group(0),  # 通信组
        is_spmd_mesh=False,       # 不是 SPMD 网格
    )
```

`FSDPMeshInfo` 封装了 FSDP 所需的全部网格信息。

```python
    # Step 4: 设备获取
    device = _get_device_from_mesh(mesh)  # → torch.device("cuda")

    # Step 5: reshard_after_forward 处理
    auto_reshard_after_forward = reshard_after_forward is None  # True（用户未指定）

    # 对于 FSDP (非 DDP) 网格：
    post_forward_mesh_info = _get_post_forward_mesh_info(
        reshard_after_forward=True,  # auto 时默认 True
        mesh_info,
    )
```

#### Step 5 详解：`_get_post_forward_mesh_info`

```python
# _fsdp_init.py
def _get_post_forward_mesh_info(reshard_after_forward, mesh_info):
    if isinstance(reshard_after_forward, bool):
        if reshard_after_forward:
            return mesh_info  # 直接返回 FSDPMeshInfo — 前向后恢复到全分片
        else:
            return None       # None — 前向后保持完整参数（不 reshard）
    elif isinstance(reshard_after_forward, int):
        # 部分 reshard：创建较小的子网格
        # 例如 reshard_after_forward=2 → 在 2 个 GPU 间分片
        ...
```

对于当前调用，`post_forward_mesh_info = mesh_info`（前向后恢复全分片）。

```python
    # Step 6: 模块和参数收集
    arg_module, modules, managed_modules, params, buffers = _get_modules_and_states(
        module, device, ignored_params
    )
```

#### Step 6 详解：`_get_modules_and_states`（关键！）

这是自底向上调用中最重要的逻辑。它决定哪些参数被当前 `fully_shard` 管理：

```python
# _fsdp_init.py
def _get_modules_and_states(module, device, ignored_params=None, ...):
    # 1. 收集所有被管理的子模块
    root_modules = (module,)  # 传入的单个 module 或 list
    managed_modules = _get_managed_modules(root_modules, ignored_params)

    # _get_managed_modules 递归遍历模块树，收集所有尚未被 FSDP 管理的模块
    # 对于 model.block0：
    #   - block0 本身：尚未被 FSDP 管理 → 纳入
    #   - block0.fc1：尚未被 FSDP 管理 → 纳入
    #   - block0.fc2：尚未被 FSDP 管理 → 纳入

    # 2. 收集所有被管理模块的参数和 buffer
    params, buffers = _get_managed_states(managed_modules, ignored_params)

    # 对于 model.block0 第一次调用：
    #   params = [block0.fc1.weight, block0.fc1.bias, block0.fc2.weight, block0.fc2.bias]
    #   buffers = [] (BatchNorm 没有)

    # 3. 如果参数已在目标设备上，跳过移动
    _move_states_to_device(params, buffers, device)

    # 4. 返回
    return arg_module, modules, managed_modules, params, buffers
```

对于第一次调用 `fully_shard(model.block0, mesh=mesh)`：

```text
arg_module      = model.block0 (Block 实例)
modules         = (model.block0,)
managed_modules = [model.block0, model.block0.fc1, model.block0.fc2]
params          = [fc1.weight (512×512), fc1.bias (512), fc2.weight (512×512), fc2.bias (512)]
buffers         = []
```

```python
    # Step 7: 初始化 FSDPState
    state = fully_shard.state(modules[0])  # 获取 @contract 附加的 state
    state.init(modules, device, mp_policy, auto_reshard_after_forward)
```

#### Step 7 详解：`FSDPState.init()`

```python
# _fsdp_state.py
class FSDPState(_State):
    def init(self, modules, device, mp_policy, auto_reshard_after_forward):
        # 7.1 将 state 关联到模块
        self._insert_module_state(modules[0])  # 通过 @contract 机制存储

        # 7.2 保存基本属性
        self._modules = modules          # (model.block0,)
        self._device = device            # torch.device("cuda")
        self._mp_policy = mp_policy      # MixedPrecisionPolicy()
        self._auto_reshard_after_forward = auto_reshard_after_forward  # True

        # 7.3 注册 forward hook
        if len(modules) == 1:
            # 单模块：直接注册 pre/post forward hook
            modules[0].register_forward_pre_hook(
                self._pre_forward, with_kwargs=True
            )
            modules[0].register_forward_hook(
                self._post_forward, always_call=True
            )
        else:
            # 多模块：使用分组 hook 注册
            _register_group_forward_hooks(
                modules, self._pre_forward, self._post_forward, ...
            )
```

**关键点**：此时注册的是 PyTorch 标准的 `register_forward_pre_hook` 和 `register_forward_hook`。当 `block0.forward()` 被调用时：
>
- `_pre_forward` 在 `block0.forward()` **之前**执行
- `_post_forward` 在 `block0.forward()` **之后**执行

```python
    # Step 8: 初始化参数组（核心步骤）
    _init_param_group(
        state, params, modules, mesh_info, post_forward_mesh_info,
        device, shard_placement_fn, mp_policy, offload_policy,
        reshard_after_forward=True,
    )
```

#### Step 8 详解：`_init_param_group`（最核心！）

```python
# _fsdp_init.py
def _init_param_group(state, params, modules, mesh_info, ...):
    # 8.1 创建 FSDPParamGroup
    state._fsdp_param_groups.append(
        FSDPParamGroup(
            params=params,              # [fc1.weight, fc1.bias, fc2.weight, fc2.bias]
            modules=modules,            # (model.block0,)
            mesh_info=mesh_info,        # FSDPMeshInfo
            post_forward_mesh_info=post_forward_mesh_info,  # FSDPMeshInfo
            device=device,              # cuda
            shard_placement_fn=None,    # 使用默认 Shard(0)
            mp_policy=MixedPrecisionPolicy(),
            offload_policy=OffloadPolicy(),
        )
    )
```

**`FSDPParamGroup.__init__` 内部**：

```python
# _fsdp_param_group.py
class FSDPParamGroup:
    def __init__(self, params, modules, mesh_info, ...):
        # 8.1.1 为每个参数创建 FSDPParam
        self.fsdp_params = [
            FSDPParam(
                param=p,
                module_info=ParamModuleInfo(
                    module=<包含该参数的模块>,
                    param_name="weight" | "bias",
                ),
                mesh_info=mesh_info,
                post_forward_mesh_info=post_forward_mesh_info,
                device=device,
                shard_placement_fn=shard_placement_fn,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
            )
            for p in params
        ]

        # 8.1.2 初始化通信上下文
        self.comm_ctx = FSDPCommContext()
        # FSDPCommContext 创建 3 条 CUDA stream：
        #   - all_gather_copy_in_stream (priority=-1, 高优先级)
        #   - all_gather_stream (priority=-1, 高优先级)
        #   - reduce_scatter_stream (priority=-1, 高优先级)
        #   - all_reduce_stream (priority=0)
        #   - post_reduce_stream (priority=0)

        # 8.1.3 设置通信器
        self._all_gather_comm = DefaultAllGather()     # 默认 NCCL all-gather
        self._reduce_scatter_comm = DefaultReduceScatter()  # 默认 NCCL reduce-scatter

        # 8.1.4 标志位初始化
        self.reduce_grads = True       # 是否执行 reduce-scatter
        self.all_reduce_grads = True   # 是否执行 all-reduce（HSDP）
        self.reshard_after_backward = True
        self.unshard_in_backward = True
```

**`FSDPParam.__init__` 内部**（每个参数的分片初始化）：

```python
# _fsdp_param.py
class FSDPParam:
    def __init__(self, param, module_info, mesh_info, ...):
        # 8.1.1a: 初始化分片参数
        self._init_sharded_param(param, device, shard_placement_fn, mesh_info)

    def _init_sharded_param(self, param, device, shard_placement_fn, mesh_info):
        # === 确定分片方式 ===
        self.fsdp_placement = Shard(0)  # 默认沿 dim=0 分片

        # === 确定分片网格的 rank 和 world_size ===
        shard_world_size = mesh_info.shard_mesh_size  # 4
        shard_rank = mesh_info.shard_mesh_rank        # 当前 GPU rank

        # === 对参数执行切分 ===
        # 以 fc1.weight (512, 512) 为例：
        param_data = param.data  # shape (512, 512)，在 CUDA 上

        # 沿 dim=0 分成 4 份
        chunks = torch.chunk(param_data, shard_world_size, dim=0)
        # chunks[0]: (128, 512)  — rank0 的分片
        # chunks[1]: (128, 512)  — rank1 的分片
        # chunks[2]: (128, 512)  — rank2 的分片
        # chunks[3]: (128, 512)  — rank3 的分片

        sharded_param = chunks[shard_rank]  # 当前 rank 的分片

        # === 填充对齐 ===
        # 如果参数不能被均匀分配，创建零填充的分片
        padded_sharded_size = chunks[0].size()  # (128, 512)
        padded_sharded_param = param_data.new_zeros(padded_sharded_size)

        # 将当前 rank 的分片复制到填充后的 tensor
        if sharded_param.numel() > 0:
            padded_sharded_param.narrow(
                dim=0, start=0, length=sharded_param.size(0)
            ).copy_(sharded_param)

        # === 展平为 1D ===
        self._sharded_param_data = padded_sharded_param.view(-1)
        # shape: (128 * 512,) = (65536,)

        # === 包装为 DTensor ===
        self.sharded_param = nn.Parameter(
            self.to_sharded_dtensor(padded_sharded_param),
            requires_grad=param.requires_grad,
        )
        # to_sharded_dtensor 内部：
        #   将 local tensor 包装为 DTensor，placements=(Shard(0),)，mesh=DeviceMesh((4,))

        # === 将分片参数注册到模块上 ===
        self._setattr_on_modules(self.sharded_param)
        # 例如：model.block0.fc1.weight 现在指向分片后的 DTensor((128, 512))

        sharded_state = ShardedState.SHARDED
```

#### 参数分片结果总结

| 参数 | 原始 shape | 分片后 shape (per rank) | 分片维度 |
| ------ | ----------- | ------------------------ | --------- |
| `block0.fc1.weight` | (512, 512) | (128, 512) | dim=0 |
| `block0.fc1.bias` | (512,) | (128,) | dim=0 |
| `block0.fc2.weight` | (512, 512) | (128, 512) | dim=0 |
| `block0.fc2.bias` | (512,) | (128,) | dim=0 |

> **注意**：bias 只有 1 维，沿 dim=0 分片意味着每个 rank 获得 bias 的一段。在 all-gather 时恢复完整 bias。

```python
    # Step 9: 标记为 FSDP 管理
    for managed_module in managed_modules:
        managed_module._is_fsdp_managed_module = True
        managed_module._fsdp_use_orig_params = True

    # Step 10: MRO 替换（核心设计！）
    _apply_to_module(
        modules, cls_to_fsdp_cls, FSDPModule, "FSDP", _unimplemented_deepcopy
    )
```

#### Step 10 详解：`_apply_to_module` — MRO 替换

```python
# _fsdp_init.py
def _apply_to_module(modules, cls_to_wrapper_cls, wrapper_module_cls,
                     wrapper_cls_prefix, unimplemented_deepcopy):
    for module in modules:
        cls = module.__class__  # Block

        # 动态创建新类：FSDPBlock extends (FSDPModule, Block)
        if cls not in cls_to_wrapper_cls:
            new_cls = type(
                f"{wrapper_cls_prefix}{cls.__name__}",  # "FSDPBlock"
                (wrapper_module_cls, cls),  # (FSDPModule, Block)
                {"__deepcopy__": unimplemented_deepcopy},
            )
            cls_to_wrapper_cls[cls] = new_cls

        # 直接修改实例的 class！
        module.__class__ = cls_to_wrapper_cls[cls]
```

修改后的 `model.block0` 的 MRO：

```text
FSDPBlock → FSDPModule → Block → nn.Module → ...
```

**FSDPModule.__new__ 的作用**：

```python
class FSDPModule:
    _orig_cls_mro_index: int = 2  # 原始类在 MRO 中的索引

    def __new__(cls, *args, **kwargs):
        # 当通过索引访问容器模块时（如 model[0]），
        # PyTorch 可能尝试用 FSDPBlock(...) 创建新实例
        # 这里拦截并直接构造原始 Block
        orig_cls = cls.__mro__[cls._orig_cls_mro_index]  # Block
        self = orig_cls.__new__(orig_cls, *args, **kwargs)
        if _enable_fsdp_module_new_init:
            self.__init__(*args, **kwargs)
        return self
```

### 2.2 第一次调用后 `model.block0` 的状态

```text
model.block0.__class__              → FSDPBlock (动态生成)
model.block0._get_fsdp_state()      → FSDPState (通过 @contract 附加)
model.block0.fc1.weight             → DTensor(shape=(128,512), placements=(Shard(0),))
model.block0.fc1.bias               → DTensor(shape=(128,), placements=(Shard(0),))
model.block0.fc2.weight             → DTensor(shape=(128,512), placements=(Shard(0),))
model.block0.fc2.bias               → DTensor(shape=(128,), placements=(Shard(0),))

FSDPState._fsdp_param_groups[0]:
  └─ FSDPParamGroup
       ├─ fsdp_params: [FSDPParam(fc1.weight), FSDPParam(fc1.bias),
       │                 FSDPParam(fc2.weight), FSDPParam(fc2.bias)]
       ├─ sharded_state: SHARDED
       ├─ post_forward_mesh_info: FSDPMeshInfo(shard_mesh_size=4)
       └─ comm_ctx: FSDPCommContext (3 条 CUDA stream)
```

---

## 3. 第二次调用：`fully_shard(model.block1, mesh=mesh)`

执行完全相同于第一次调用的流程，但操作对象是 `model.block1`：

```text
arg_module      = model.block1 (Block 实例)
modules         = (model.block1,)
managed_modules = [model.block1, model.block1.fc1, model.block1.fc2]
params          = [block1.fc1.weight, block1.fc1.bias, block1.fc2.weight, block1.fc2.bias]

MRO 替换后: model.block1.__class__ = FSDPBlock
分片后:     block1.fc1.weight = DTensor(128, 512)
```

此时 `model.block0` 和 `model.block1` 都有各自独立的：
>
- `FSDPState`（通过 `@contract` 附加）
- `FSDPParamGroup`（管理各自的 4 个参数）
- `FSDPCommContext`（独立的 CUDA stream）

---

## 4. 第三次调用：`fully_shard(model, mesh=mesh)` — 顶层包装

这是最关键的调用。顶层包装与子模块包装有显著区别。

### 4.1 `_get_modules_and_states` 的行为差异

```python
# _fsdp_init.py
def _get_managed_modules(root_modules, ignored_params=None, ...):
    # 递归遍历 module 的所有子模块
    # 关键：跳过已经被 FSDP 管理的模块！
    for module in root_modules:
        if not isinstance(module, FSDPModule):  # ← 跳过已 FSDP 包装的模块
            # 但遍历其子模块
            for child in module.children():
                ...
```

对于 `model`：
>
- `model` 本身不是 `FSDPModule` → 纳入管理
- `model.block0` 已经是 `FSDPModule` → **跳过**（其参数已被 block0 的 FSDPParamGroup 管理）
- `model.block1` 已经是 `FSDPModule` → **跳过**
- `model.block0.fc1` 已被 `_is_fsdp_managed_module` 标记 → 跳过
- `model.block1.fc2` 同理 → 跳过

**结果**：

```text
managed_modules = [model]  # 只有 model 本身（无参数）
params          = []       # 空！所有参数已被子模块的 fully_shard 管理
modules         = (model,)
```

### 4.2 顶层参数组初始化

由于 `params = []`，创建的 `FSDPParamGroup` 没有 `FSDPParam`：

```python
state._fsdp_param_groups.append(
    FSDPParamGroup(
        params=[],           # 空列表！
        modules=(model,),
        ...
    )
)
```

但 `FSDPModule` 的 MRO 替换仍然发生：

```python
model.__class__ = FSDPModel  # 新类名
# MRO: FSDPModel → FSDPModule → Model → nn.Module → ...
```

### 4.3 顶层模块的特殊性

顶层 `model` 的 `FSDPState` 将扮演"根状态"角色，承担以下关键职责：

1. **延迟初始化触发器**：首次 `_pre_forward` 时执行 `_lazy_init()`
2. **Stream 同步**：等待上次反向的通信操作完成
3. **输入数据移动**：将输入 tensor 移动到正确的 CUDA 设备
4. **最终回调注册**：通过 autograd 引擎注册 `_root_post_backward_final_callback`

### 4.4 三次 fully_shard 后的完整状态

```text
model (FSDPModel)
├── __class__ = FSDPModel (MRO: FSDPModel→FSDPModule→Model→nn.Module)
├── FSDPState (根状态, _is_root=True — lazily set)
│   └── FSDPParamGroup (空参数组)
│
├── block0 (FSDPBlock)
│   ├── __class__ = FSDPBlock (MRO: FSDPBlock→FSDPModule→Block→nn.Module)
│   ├── FSDPState (子状态, _is_root=False)
│   │   └── FSDPParamGroup
│   │       ├── FSDPParam(fc1.weight) → DTensor(128,512)
│   │       ├── FSDPParam(fc1.bias)   → DTensor(128,)
│   │       ├── FSDPParam(fc2.weight) → DTensor(128,512)
│   │       └── FSDPParam(fc2.bias)   → DTensor(128,)
│   │
│   ├── fc1.weight → DTensor((128, 512), placements=(Shard(0),))
│   ├── fc1.bias   → DTensor((128,), placements=(Shard(0),))
│   ├── fc2.weight → DTensor((128, 512), placements=(Shard(0),))
│   └── fc2.bias   → DTensor((128,), placements=(Shard(0),))
│
└── block1 (FSDPBlock)
    ├── __class__ = FSDPBlock
    ├── FSDPState (子状态, _is_root=False)
    │   └── FSDPParamGroup
    │       ├── FSDPParam(fc1.weight) → DTensor(128,512)
    │       ├── FSDPParam(fc1.bias)   → DTensor(128,)
    │       ├── FSDPParam(fc2.weight) → DTensor(128,512)
    │       └── FSDPParam(fc2.bias)   → DTensor(128,)
    │
    ├── fc1.weight → DTensor((128, 512), placements=(Shard(0),))
    ├── fc1.bias   → DTensor((128,), placements=(Shard(0),))
    ├── fc2.weight → DTensor((128, 512), placements=(Shard(0),))
    └── fc2.bias   → DTensor((128,), placements=(Shard(0),))
```

---

## 5. 前向计算：`output = model(x)` — 完整追踪

当调用 `model(x)` 时，由于 MRO 替换，实际调用链如下。注意 FSDP2 并非通过重写 `forward` 方法实现拦截，而是通过**注册 forward pre-hook 和 forward hook**：

```text
model.__call__(x)                         # nn.Module.__call__
  └─ 触发 registered forward pre-hooks:
       └─ FSDPState._pre_forward(model, args, kwargs)    ← 根前向
            ├─ _root_pre_forward()
            │    ├─ _lazy_init()            ← 延迟初始化（首次调用）
            │    └─ Stream 同步
            └─ FSDPParamGroup.pre_forward() (空操作，无参数)

  └─ model.forward(x)                       # Model.forward
       └─ self.block0(x)                    # 触发 block0 的 __call__
            └─ 触发 block0 的 forward pre-hook:
                 └─ FSDPState._pre_forward(block0, args, kwargs)
                      ├─ block0.FSDPParamGroup.pre_forward():
                      │    ├─ unshard()         ← all-gather 参数
                      │    └─ wait_for_unshard() ← 等待通信完成
                      └─ _prefetch_unshard(block1, "forward")  ← 预取

            └─ block0.forward(x)             # Block.forward — 实际计算
                 └─ fc1(x) → relu → fc2(x)

            └─ 触发 block0 的 forward hook:
                 └─ FSDPState._post_forward(block0, args, output)
                      └─ block0.FSDPParamGroup.post_forward()
                           └─ reshard()       ← 释放完整参数

       └─ self.block1(x)                    # 触发 block1 的 __call__
            └─ (同样的 pre_forward → forward → post_forward 流程)

       └─ return x

  └─ 触发 model 的 forward hook:
       └─ FSDPState._post_forward(model, args, output)
```

### 5.1 第一阶段：`model` 的 `_pre_forward`（根前向预处理）

```python
# _fsdp_state.py: FSDPState._pre_forward()
def _pre_forward(self, module, args, kwargs):
    # 检查是否为重计算场景（SAC activation checkpointing）
    if self._training_state == TrainingState.PRE_BACKWARD:
        # 重计算：参数可能尚未 unshard，手动触发
        for fsdp_param_group in self._fsdp_param_groups:
            if not fsdp_param_group.is_unsharded:
                fsdp_param_group.unshard()
                fsdp_param_group.wait_for_unshard()
        return args, kwargs

    # 正常前向
    self._training_state = TrainingState.FORWARD
    self._root_pre_forward(module, args, kwargs)  # 根前向同步

    # 输入精度转换（如果设置了 mp_policy.param_dtype）
    args, kwargs = _cast_fp_tensor(args, kwargs, self._mp_policy.param_dtype)

    # 执行参数组的 pre_forward
    for fsdp_param_group in self._fsdp_param_groups:
        fsdp_param_group.pre_forward(module, args, kwargs)

    # 前向预取
    for target_state in self._states_to_forward_prefetch:
        for target_group in target_state._fsdp_param_groups:
            FSDPParamGroup._prefetch_unshard(target_group, "forward")

    return args, kwargs
```

#### 5.1.1 根前向同步：`_root_pre_forward`

```python
# _fsdp_state.py
def _root_pre_forward(self, module, args, kwargs):
    # === 场景检查: 防止重入 ===
    if self._training_state in (TrainingState.FORWARD, TrainingState.PRE_BACKWARD):
        return  # 已经在 forward 或 pre-backward 状态

    # === 延迟初始化（首次调用时） ===
    self._lazy_init(module)

    # === Stream 同步: 确保上次 optimizer step 已完成 ===
    if self._state_ctx.post_optim_event is not None:
        # 用户提供了 post-optimizer event
        self._comm_ctx.all_gather_copy_in_stream.wait_event(
            self._state_ctx.post_optim_event
        )
        self._comm_ctx.all_gather_stream.wait_event(
            self._state_ctx.post_optim_event
        )
        self._state_ctx.post_optim_event = None
    else:
        # 默认：当前流等待 all-gather 流
        # 确保上次反向的 reduce-scatter 等操作已完成
        self._device_handle.current_stream().wait_stream(
            self._comm_ctx.all_gather_copy_in_stream
        )
        self._device_handle.current_stream().wait_stream(
            self._comm_ctx.all_gather_stream
        )

    # === 输入数据移动 ===
    # 对于 CUDA/HPU/XPU/MTIA，将输入 tensor 移到正确的设备
    kwargs = _to_kwargs(kwargs, self._device)

    return args, kwargs
```

#### 5.1.2 延迟初始化：`_lazy_init`（首次调用的核心！）

```python
# _fsdp_state.py
def _lazy_init(self, module):
    if self._is_root is not None:
        return  # 已经初始化过

    # === 标记根状态 ===
    self._is_root = True

    # === 遍历所有子模块，设置非根状态 ===
    fsdp_states = []
    for submodule in module.modules():
        if isinstance(submodule, FSDPModule):
            state = submodule._get_fsdp_state()
            if state is not self:
                state._is_root = False
            fsdp_states.append(state)

    # === 共享 StateContext 和 CommContext ===
    # 所有子状态共享根的 context，避免重复创建 CUDA stream
    self._state_ctx.all_states = fsdp_states
    for state in fsdp_states:
        state._state_ctx = self._state_ctx
        state._comm_ctx = self._comm_ctx

    # === auto_reshard_after_forward 处理 ===
    # 如果不自动 reshard，设置 post_forward_mesh_info = None
    # 这意味着前向后参数保持完整（不释放），后续前向无需再次 all-gather
    if self._auto_reshard_after_forward:
        for state in fsdp_states:
            for fsdp_param_group in state._fsdp_param_groups:
                # 根模块特殊处理：
                # 如果当前 group 的模块是根模块 → post_forward_mesh_info = None
                # 这是性能优化：根模块通常紧接着被反向使用
                if fsdp_param_group._is_root_module(module):
                    fsdp_param_group.post_forward_mesh_info = None

    # === 验证参数无重复 ===
    self._validate_no_duplicate_params()

    # === 执行各参数组的 lazy_init ===
    for state in fsdp_states:
        for fsdp_param_group in state._fsdp_param_groups:
            fsdp_param_group.lazy_init()
```

**`FSDPParamGroup.lazy_init()`**：

```python
# _fsdp_param_group.py
def lazy_init(self):
    # 确保 device_handle 可用
    if self.comm_ctx is not None and self.comm_ctx.device_handle is None:
        self.comm_ctx.lazy_init(self.device)

    # 如果已分片但参数需要重置（如从 checkpoint 加载后）
    if self.is_sharded and not self._reset_sharded_params:
        for fsdp_param in self.fsdp_params:
            fsdp_param.reset_sharded_param()
        self._init_extensions()
        self._reset_sharded_params = True

    # 验证：无 meta 设备参数
    self._validate_no_meta_params()

    # 初始化混合精度 dtype
    self._init_mp_dtypes()

    # 注册 state_dict hook
    self._register_state_dict_hooks()
```

#### 5.1.3 根参数组的 `pre_forward`

由于 `model` 的参数组没有参数（`params = []`），`pre_forward` 本质上是空操作：

```python
# FSDPParamGroup.pre_forward():
def pre_forward(self, module, args, kwargs):
    self._training_state = TrainingState.FORWARD
    self.unshard()           # 无参数 → 直接返回
    self.wait_for_unshard()  # 无参数 → 直接返回
    self._register_post_backward_hook(args, kwargs)  # 注册反向 hook
    return args, kwargs
```

### 5.2 第二阶段：`model.forward(x)` 原始前向

```python
# Model.forward()
def forward(self, x):
    x = self.block0(x)   # 触发 block0 的 __call__
    x = self.block1(x)   # 触发 block1 的 __call__
    return x
```

#### 5.2.1 `block0.__call__(x)` 触发 block0 的 `_pre_forward`

```python
# FSDPState._pre_forward(block0, args, kwargs)
def _pre_forward(self, module, args, kwargs):
    self._training_state = TrainingState.FORWARD
    self._root_pre_forward(module, args, kwargs)  # 非根模块：无操作

    # 输入精度转换
    args, kwargs = _cast_fp_tensor(args, kwargs, self._mp_policy.param_dtype)

    # === block0 的参数组 pre_forward ===
    for fsdp_param_group in self._fsdp_param_groups:
        fsdp_param_group.pre_forward(module, args, kwargs)
        # ↑ 这里执行 all-gather + wait，恢复完整参数
```

**`block0.FSDPParamGroup.pre_forward()` 详解**：

```python
# _fsdp_param_group.py
def pre_forward(self, module, args, kwargs):
    # 1. 设置状态
    self._training_state = TrainingState.FORWARD

    # 2. 触发 all-gather（异步）
    self.unshard()

    # 3. 等待 all-gather 完成
    self.wait_for_unshard()

    # 4. 注册反向 hook（在输出张量上）
    self._register_post_backward_hook(args, kwargs)

    return args, kwargs
```

##### 5.2.1.1 `unshard()` — 异步 All-Gather

```python
# _fsdp_param_group.py
def unshard(self, async_op=False):
    if self.is_unsharded:
        return  # 已经是完整参数，跳过

    if self.world_size == 1:
        self._handle = AllGatherState(None, None, None)  # 单 GPU 无需通信
        return

    # === 收集所有参数的分片数据 ===
    all_gather_inputs = []
    all_gather_numels = []
    for fsdp_param in self.fsdp_params:
        inputs = fsdp_param.all_gather_inputs  # 返回 1D 分片数据
        all_gather_inputs.extend(inputs)
        all_gather_numels.append(inputs[0].numel())

    # === 异步执行 all-gather（双流流水线） ===
    self._handle = foreach_all_gather(
        all_gather_inputs,
        all_gather_numels,
        self.fsdp_params,        # [FSDPParam(fc1.weight), FSDPParam(fc1.bias), ...]
        group=self.mesh_info.shard_process_group,  # 4 GPU 通信组
        all_gather_comm=self._all_gather_comm,
        copy_in_stream=self.comm_ctx.all_gather_copy_in_stream,
        all_gather_stream=self.comm_ctx.all_gather_stream,
        async_op=async_op,
    )
```

**`foreach_all_gather` 双流流水线**：

```python
# _fsdp_collectives.py
def foreach_all_gather(fsdp_params, group, async_op,
                        all_gather_copy_in_stream, all_gather_stream,
                        device, all_gather_comm):
    world_size = group.size()  # 4
    rank = group.rank()
    device_handle = _get_device_handle(device.type)

    # ====== 阶段 1: Copy-In (在 all_gather_copy_in_stream 上) ======
    with device_handle.stream(all_gather_copy_in_stream):
        # 1.1 收集所有分片输入
        all_gather_inputs = _get_param_all_gather_inputs(fsdp_params)
        # 返回: [fc1.weight_shard_1D, fc1.bias_shard_1D, fc2.weight_shard_1D, fc2.bias_shard_1D]

        # 1.2 计算总大小
        all_gather_input_numel = sum(inp.numel() for inp in all_gather_inputs)
        # = 128*512 + 128 + 128*512 + 128 = 131200

        # 1.3 分配输出缓冲区
        # all_gather_output shape: (131200 * 4,) = (524800,)
        all_gather_output = all_gather_comm.allocate(
            all_gather_input_numel * world_size
        )

        # 1.4 执行 copy-in：将每个 rank 的分片数据复制到输出缓冲区的对应位置
        # rank=0 的数据 → output[0:131200]
        # rank=1 的数据 → output[131200:262400]
        # rank=2 的数据 → output[262400:393600]
        # rank=3 的数据 → output[393600:524800]
        torch.ops.fsdp.all_gather_copy_in(
            all_gather_inputs,
            all_gather_output,
            inp_split_sizes,
            all_gather_input_numel,
            rank,
        )

    # ====== 同步: all_gather_stream 等待 copy_in_stream ======
    all_gather_stream.wait_stream(all_gather_copy_in_stream)

    # ====== 阶段 2: All-Gather (在 all_gather_stream 上) ======
    with device_handle.stream(all_gather_stream):
        # 执行 NCCL all-gather
        all_gather_work = all_gather_comm(all_gather_output, group)
        # 等价于: dist.all_gather_into_tensor(all_gather_output, ..., group=group)

        # 记录事件用于后续同步
        all_gather_event = all_gather_stream.record_event()

    return AllGatherResult(
        all_gather_output=all_gather_output,
        all_gather_event=all_gather_event,
        all_gather_work=all_gather_work,
        all_gather_input_dtypes=...,
        all_gather_input_numels=all_gather_numels,
        ...
    )
```

##### 5.2.1.2 `wait_for_unshard()` — 等待通信完成

```python
# _fsdp_param_group.py
def wait_for_unshard(self):
    all_gather_result = self._handle  # AllGatherResult

    if self.world_size == 1:
        # 单 GPU：直接复制分片参数（不需要通信）
        for fsdp_param in self.fsdp_params:
            fsdp_param.init_all_gather_outputs()
            fsdp_param.alloc_all_gather_outputs()
            fsdp_param.init_unsharded_param()
        return

    # === 执行 copy-out ===
    foreach_all_gather_copy_out(
        all_gather_result,
        self.fsdp_params,
        group=self.mesh_info.shard_process_group,
    )
```

**`foreach_all_gather_copy_out` 详解**：

```python
# _fsdp_collectives.py
def foreach_all_gather_copy_out(all_gather_result, fsdp_params, group, ...):
    all_gather_output, all_gather_event, all_gather_work, ... = all_gather_result

    # === 1. 同步：当前流等待 all-gather 完成 ===
    if all_gather_event is not None:
        current_stream.wait_event(all_gather_event)
    if all_gather_work is not None:
        all_gather_work.wait()

    # === 2. 分配合并初始化每个参数的非分片输出 ===
    for fsdp_param in fsdp_params:
        fsdp_param.init_all_gather_outputs()
        fsdp_param.alloc_all_gather_outputs()

    # === 3. 将 all-gather 输出拆分到各参数的输出缓冲区 ===
    torch.ops.fsdp.split_with_sizes_copy(
        all_gather_output, all_gather_output_splits, ...
    )

    # === 4. 初始化非分片参数 ===
    for fsdp_param in fsdp_params:
        fsdp_param.init_unsharded_param()
```

**`FSDPParam.init_unsharded_param()`**：

```python
# _fsdp_param.py
def init_unsharded_param(self):
    # from all_gather_outputs[0], shape: (512*512,) = (262144,) — 完整 fc1.weight 的 1D 视图

    # 使用 as_strided 创建完整参数的视图（零拷贝，共享存储）
    unsharded_tensor = torch.as_strided(
        self.all_gather_outputs[0],
        size=self._orig_size,               # (512, 512)
        stride=self._contiguous_orig_stride,  # (512, 1)
        storage_offset=0,
    )

    # 包装为 nn.Parameter
    self._unsharded_param = nn.Parameter(
        unsharded_tensor,
        requires_grad=self.sharded_param.requires_grad,
    )

    # 将完整参数注册到模块
    # model.block0.fc1.weight 现在指向完整的 (512, 512) Tensor！
    self._setattr_on_modules(self._unsharded_param)

    self.sharded_state = ShardedState.UNSHARDED
```

**block0 unshard 后的状态变化**：

| 参数 | unshard 前 | unshard 后 |
| ------ | ----------- | ----------- |
| `block0.fc1.weight` | DTensor(128, 512) | Tensor(512, 512) — 完整参数 |
| `block0.fc1.bias` | DTensor(128,) | Tensor(512,) — 完整 bias |
| `block0.fc2.weight` | DTensor(128, 512) | Tensor(512, 512) — 完整参数 |
| `block0.fc2.bias` | DTensor(128,) | Tensor(512,) — 完整 bias |

##### 5.2.1.3 前向预取：block1 的 all-gather

```python
# 在 block0 的 _pre_forward 返回前：
for target_state in self._states_to_forward_prefetch:
    for target_group in target_state._fsdp_param_groups:
        FSDPParamGroup._prefetch_unshard(target_group, "forward")
```

由于当前 `_states_to_forward_prefetch` 为空（默认情况），预取通过**隐式方式**进行。实际上，预取在 `wait_for_unshard` 中被触发（源码中使用 `_handle` 保存前一个 all-gather 状态进行处理）。

**默认隐式预取**的逻辑是：FSDP2 利用 `wait_for_unshard` 中保存的前一次 `AllGatherResult`，在合适时机释放其内存，同时触发下一个模块的 unshard。

#### 5.2.2 `block0.forward(x)` — 实际计算

```python
# Block.forward
def forward(self, x):           # x: (32, 512)
    x = self.fc1(x)            # fc1.weight: (512, 512) — 完整参数
                               # fc1.bias: (512,) — 完整 bias
                               # → x: (32, 512)
    x = torch.relu(x)          # → x: (32, 512)
    x = self.fc2(x)            # fc2.weight: (512, 512) — 完整参数
                               # fc2.bias: (512,) — 完整 bias
                               # → x: (32, 512)
    return x
```

此时 `fc1.weight`, `fc1.bias`, `fc2.weight`, `fc2.bias` 都是完整的（已被 `unshard()` 恢复），因此计算和普通的 PyTorch 模块完全一致。

#### 5.2.3 `block0` 的 `_post_forward` — 释放完整参数

```python
# _fsdp_state.py
def _post_forward(self, module, args, output):
    if self._training_state == TrainingState.PRE_BACKWARD:
        return output  # 重计算场景，不释放

    for fsdp_param_group in self._fsdp_param_groups:
        fsdp_param_group.post_forward()

    # 注册 pre-backward hook（在输出 tensor 上）
    self._register_pre_backward_hook(output)

    self._training_state = TrainingState.IDLE
    return output
```

**`FSDPParamGroup.post_forward()`**：

```python
# _fsdp_param_group.py
def post_forward(self):
    self.reshard()  # ← 释放完整参数，恢复分片参数
    self._training_state = TrainingState.IDLE
```

**`FSDPParamGroup.reshard()` 内部**：

```python
# _fsdp_param_group.py
def reshard(self):
    if self.post_forward_mesh_info is not None:
        # 有 post_forward mesh → reshard 到 post_forward 粒度
        self._to_sharded_post_forward()
    else:
        # 无 post_forward mesh → reshard 回原始分片
        self._to_sharded()
```

**`_to_sharded()`**：

```python
# _fsdp_param_group.py
def _to_sharded(self):
    for fsdp_param in self.fsdp_params:
        fsdp_param.to_sharded()  # 每个参数单独处理
    self._sharded_state = ShardedState.SHARDED
```

**`FSDPParam.to_sharded()`**：

```python
# _fsdp_param.py
def to_sharded(self):
    # 将分片参数重新注册到模块
    self._setattr_on_modules(self.sharded_param)
    # model.block0.fc1.weight 现在又变回 DTensor(128, 512)

    # 释放非分片参数的存储
    self.free_unsharded_param()
    # 释放 all_gather_outputs 的底层存储
    # 内存峰值降低！

    self.sharded_state = ShardedState.SHARDED
```

**`free_unsharded_param()`**：

```python
# _fsdp_param.py
def free_unsharded_param(self):
    # 释放 all-gather 输出缓冲区
    for t in self.all_gather_outputs:
        free_storage(t)  # storage.resize_(0)

    # 释放扩展相关的内部 tensor（如有）
    if self._extensions_data:
        self._extensions_data.clear()
```

**block0 reshard 后**：

| 参数 | reshard 前 | reshard 后 |
| ------ | ----------- | ----------- |
| `block0.fc1.weight` | Tensor(512, 512) | DTensor(128, 512) |
| `block0.fc1.bias` | Tensor(512,) | DTensor(128,) |
| `block0.fc2.weight` | Tensor(512, 512) | DTensor(128, 512) |
| `block0.fc2.bias` | Tensor(512,) | DTensor(128,) |

内存中 all-gather 的输出缓冲区已被释放，GPU 显存占用大幅降低。

#### 5.2.4 `block1.__call__(x)` — 重复相同流程

`block1` 的 `_pre_forward` → `forward()` → `_post_forward` 流程与 `block0` 完全相同：

1. **pre_forward**: unshard（all-gather 参数）→ wait_for_unshard → 注册反向 hook
2. **forward**: fc1 → relu → fc2（使用完整参数）
3. **post_forward**: reshard（释放完整参数）

#### 5.2.5 `model` 的 `_post_forward`

```python
# FSDPState._post_forward(model, args, output)
def _post_forward(self, module, args, output):
    for fsdp_param_group in self._fsdp_param_groups:
        fsdp_param_group.post_forward()  # 空操作（无参数）

    # 注册 pre-backward hook
    self._register_pre_backward_hook(output)

    self._training_state = TrainingState.IDLE

    # 如果是 iter_forward_root（微批次场景），清理状态
    if self._state_ctx.iter_forward_root is self:
        self._state_ctx.iter_forward_root = None

    return output
```

---

## 6. 完整前向计算的 Stream 时序图

将整个前向过程的 CUDA stream 交互可视化：

```text
时间 →

计算流 (default stream):
  │
  ├─ model._pre_forward
  │    ├─ _lazy_init()
  │    └─ Stream 同步 (wait all_gather_stream)
  │
  ├─ model.forward(x)
  │    │
  │    ├─ block0.__call__(x)
  │    │    ├─ block0._pre_forward
  │    │    │    ├─ unshard() ───────────────────────────────────────┐
  │    │    │    ├─ wait_for_unshard()   ← 等待 all_gather event ──┐ │
  │    │    │    └─ 前向预取 block1                                  │ │
  │    │    ├─ block0.forward() [计算] (使用完整参数)                │ │
  │    │    └─ block0._post_forward                                  │ │
  │    │         └─ reshard() (释放完整参数)                         │ │
  │    │                                                             │ │
  │    ├─ block1.__call__(x)                                         │ │
  │    │    ├─ block1._pre_forward                                   │ │
  │    │    │    ├─ unshard() ────────────────────────────────────┐  │ │
  │    │    │    ├─ wait_for_unshard()  ← 等待 all_gather event ─┐│  │ │
  │    │    │    └─ 前向预取 (无下一层)                           ││  │ │
  │    │    ├─ block1.forward() [计算]                            ││  │ │
  │    │    └─ block1._post_forward                               ││  │ │
  │    │         └─ reshard()                                     ││  │ │
  │    └─ return                                                  ││  │ │
  │                                                               ││  │ │
  └─ model._post_forward (注册 pre-backward hook)                 ││  │ │
                                                                  ││  │ │
all_gather_copy_in_stream (priority=-1):                           ││  │ │
  ├─ [copy_in B0] ──────────────────────────────────────────────  ││  │ │
  │                                                               ││  │ │
  └─ wait ────────────────────────────────────────────────────── ││  │ │
                                                                  ││  │ │
all_gather_stream (priority=-1):                                  ││  │ │
  ├─ [all-gather B0] ── event_B0 ─────────────────────────────── ││  │ │
  │                    (计算流等待)  ▲                             ││  │ │
  │                                 │                             ││  │ │
  ├─ [copy_in B1] ───────────────┐  │                             ││  │ │
  │                               │  │                             ││  │ │
  └─ [all-gather B1] ── event_B1   │                             ││  │ │
                       (计算流等待)─┘                             ││  │ │
                                                                  ││  │ │
图例：                                                            ││  │ │
 ── = 时间线                                                      ││  │ │
 ▲  = 同步点（wait_event）                                        ││  │ │
```

### 关键重叠说明

1. **block0 的 copy-in 和 all-gather 在独立流上**，与计算流并行
2. **block1 的 all-gather 可以在 block0 计算期间启动**（如果使用显式预取 `set_modules_to_forward_prefetch`）
3. **默认隐式预取**：`wait_for_unshard` 中保存前一次 `AllGatherResult`，在下一次 `unshard` 时释放

---

## 7. 关键源码路径速查

| 步骤 | 文件 | 函数/类 |
| ------ | ------ | --------- |
| API 入口 | `_fully_shard.py` | `fully_shard()` |
| 模块验证 | `_fsdp_init.py` | `_validate_module()` |
| Mesh 解析 | `_fsdp_init.py` | `_get_mesh_info()` |
| 模块/参数收集 | `_fsdp_init.py` | `_get_modules_and_states()` |
| State 初始化 | `_fsdp_state.py` | `FSDPState.init()` |
| 参数组创建 | `_fsdp_init.py` | `_init_param_group()` → `FSDPParamGroup.__init__()` |
| 参数分片 | `_fsdp_param.py` | `FSDPParam._init_sharded_param()` |
| MRO 替换 | `_fsdp_init.py` | `_apply_to_module()` |
| 前向 Hook | `_fsdp_state.py` | `FSDPState._pre_forward()` / `_post_forward()` |
| 延迟初始化 | `_fsdp_state.py` | `FSDPState._lazy_init()` |
| 根前向同步 | `_fsdp_state.py` | `FSDPState._root_pre_forward()` |
| All-Gather | `_fsdp_collectives.py` | `foreach_all_gather()` |
| Copy-Out | `_fsdp_collectives.py` | `foreach_all_gather_copy_out()` |
| Reshard | `_fsdp_param_group.py` | `FSDPParamGroup.reshard()` → `_to_sharded()` |
| 完整参数构建 | `_fsdp_param.py` | `FSDPParam.init_unsharded_param()` |
| 内存释放 | `_fsdp_param.py` | `FSDPParam.free_unsharded_param()` |

---

## 8. 总结：fully_shard 做了什么

以本文的 `model(x)` 为例，三次 `fully_shard` 调用和一次前向计算的本质是：

### fully_shard 阶段（构建时）

1. **参数分组**：将模块的参数打包成 `FSDPParamGroup`
2. **参数分片**：每个参数沿 dim=0 切分为 `world_size` 份，每个 rank 只保留自己的分片（DTensor）
3. **MRO 替换**：修改 `module.__class__`，将 `FSDPModule` 插入类层次结构
4. **Hook 注册**：注册 forward pre-hook 和 forward hook

### model(x) 阶段（运行时）

1. **延迟初始化**（首次调用）：共享 context、验证参数、初始化通信器
2. **Stream 同步**：确保上次反向的 reduce-scatter 已完成
3. **block0 前向**：unshard（all-gather 恢复完整参数）→ 计算 → reshard（释放完整参数）
4. **block1 前向**：unshard → 计算 → reshard
5. **反向 hook 注册**：为后续 `loss.backward()` 做好准备

### 关键性能特征

- **内存峰值**：只在当前计算层的参数 unshard 时占用完整内存，其余层保持分片
- **通信-计算重叠**：通过独立 CUDA stream 和预取机制，all-gather 与前一层计算重叠
- **零拷贝 unshard**：`torch.as_strided` 创建视图，不复制数据

---

> **源码参考**：
>
> - `fully_shard` 入口：`torch/distributed/fsdp/_fully_shard/_fully_shard.py`
> - 模块收集与参数管理：`torch/distributed/fsdp/_fully_shard/_fsdp_init.py`
> - State 管理：`torch/distributed/fsdp/_fully_shard/_fsdp_state.py`
> - 参数组管理：`torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py`
> - 参数分片：`torch/distributed/fsdp/_fully_shard/_fsdp_param.py`
> - 通信原语：`torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py`
