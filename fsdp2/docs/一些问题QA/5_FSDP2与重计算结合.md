# FSDP2 与重计算（Activation Checkpointing）结合

> 基于 PyTorch v2.12.0 源码分析
> 核心源码：
>
> - `_fsdp_state.py`：`TrainingState` 状态机 + `_pre_forward` / `_post_forward` 中的 AC 检测
> - `_fsdp_param_group.py`：`pre_backward` 中的 unshard 触发

## 0. 重计算的基本概念

Activation Checkpointing（AC，又称 gradient checkpointing 或重计算）通过**不保存前向中间激活值，反向时重新计算**来节省显存。

```text
无 AC:
  前向: 计算并保存所有中间激活 → 显存: 高
  反向: 使用保存的激活计算梯度

有 AC:
  前向: 计算但不保存 Checkpoint 区域内的激活 → 显存: 低
  反向: 重新执行 Checkpoint 区域的前向 → 重新生成激活 → 计算梯度
```

**FSDP2 + AC 的核心挑战**：

1. 前向时 FSDP 需要 all-gather 参数
2. 反向重计算时，会**再次触发 FSDP 的前向 hook**
3. 如果不处理，会导致重复 all-gather、重复 reshard、hook 重复注册

## 1. 示例代码

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.utils.checkpoint import checkpoint

dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)

mesh = init_device_mesh("cuda", (4,))

class Block(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = Block(512, 2048)
        self.block1 = Block(512, 2048)
        self.block2 = Block(512, 2048)

    def forward(self, x):
        x = self.block0(x)
        # ====== 对 block1 使用 activation checkpointing ======
        x = checkpoint(self.block1, x, use_reentrant=False)
        x = self.block2(x)
        return x

model = Model().cuda()

# 先 FSDP 包装，再 AC（推荐顺序）
for block in [model.block0, model.block1, model.block2]:
    fully_shard(block, mesh=mesh)
fully_shard(model, mesh=mesh)

x = torch.randn(32, 512).cuda()
output = model(x)
loss = output.sum()
loss.backward()  # ← AC 重计算 + FSDP 交互发生在这里
```

## 2. TrainingState 状态机：AC 交互的核心

FSDP2 使用 `TrainingState` 枚举来追踪当前处于训练的哪个阶段：

```python
# _fsdp_common.py
class TrainingState(Enum):
    IDLE = 0          # 空闲状态
    FORWARD = 1       # 前向计算中
    PRE_BACKWARD = 2  # 反向预计算中（AC 重计算也使用此状态）
    POST_BACKWARD = 3 # 反向计算完成（梯度已 reduce）
```

**关键设计**：`PRE_BACKWARD` 状态是 FSDP2 与 AC 交互的"信号量"。

## 3. AC 重计算时的 _pre_forward 行为

### 3.1 正常前向的 _pre_forward

```python
# _fsdp_state.py: FSDPState._pre_forward()
def _pre_forward(self, module, args, kwargs):
    # 正常前向路径
    if self._training_state == TrainingState.IDLE:
        # 1. 根前向预处理（stream sync, lazy init）
        self._root_pre_forward(module)
        
        # 2. 对每个参数组执行 unshard
        for fsdp_param_group in self._fsdp_param_groups:
            fsdp_param_group.pre_forward(module, args, kwargs)
            #  pre_forward 内部:
            #    training_state = FORWARD
            #    unshard() + wait_for_unshard()
            #    注册 pre-backward hook
        
        # 3. 预取下一层
        self._prefetch_forward()
        
        self._training_state = TrainingState.FORWARD
```

### 3.2 AC 重计算时的 _pre_forward

```python
# _fsdp_state.py: FSDPState._pre_forward() — AC 重计算路径
def _pre_forward(self, module, args, kwargs):
    # === AC 重计算检测 ===
    if self._training_state == TrainingState.PRE_BACKWARD:
        # 跳过根前向预处理（不需要再次 stream sync, lazy init）
        # 只确保参数是 unshard 的
        for fsdp_param_group in self._fsdp_param_groups:
            if not fsdp_param_group.is_unsharded:
                fsdp_param_group.unshard()
                fsdp_param_group.wait_for_unshard()
        # 不改变 training_state，保持 PRE_BACKWARD
        return
```

**核心逻辑**：

- 正常前向时，`_training_state` 从 `IDLE` → `FORWARD`
- AC 重计算时，`_training_state` 已经是 `PRE_BACKWARD`（由 `_pre_backward` 设置的）
- 检测到 `PRE_BACKWARD` 后，只做最小化的 unshard（如果参数还未恢复）
- **不重复执行** root 逻辑、stream sync、hook 注册

### 3.3 为什么 AC 重计算时参数可能已经 unshard

```text
时间线 (block1 有 AC):
  Forward:
    _pre_forward(block1) → unshard → FORWARD → 计算 → _post_forward(block1) → reshard → IDLE
    ...
    _post_forward(model) → 注册 pre-backward hook on output

  Backward (AC 重计算):
    loss.backward()
      ↓
    _pre_backward(block1) → PRE_BACKWARD → unshard（恢复参数）
      ↓
    checkpoint 框架：重新执行 block1.forward()
      ↓
    _pre_forward(block1) → 检测到 PRE_BACKWARD → 参数已 unshard → 跳过！
      ↓
    block1.forward() 重计算（使用已 unshard 的参数）
      ↓
    _post_forward(block1) → 检测到 PRE_BACKWARD → 跳过 reshard！
      ↓
    计算梯度
      ↓
    post_backward(block1) → reduce-scatter
```

## 4. AC 重计算时的 _post_forward 行为

```python
# _fsdp_state.py: FSDPState._post_forward()
def _post_forward(self, module, args, output):
    if self._training_state == TrainingState.PRE_BACKWARD:
        # AC 重计算场景：不做任何操作，直接返回
        # - 不 reshard（参数需要用于梯度计算）
        # - 不注册 pre-backward hook（前向时已注册）
        # - 不清理 all-gather 状态
        return output
    
    # 正常前向路径
    for fsdp_param_group in self._fsdp_param_groups:
        fsdp_param_group.post_forward()  # reshard
    self._training_state = TrainingState.IDLE
    return output
```

**为什么重计算时不能 reshard？**

因为参数马上就要用于梯度计算。如果 reshard，梯度计算时拿不到完整参数。

**为什么不能重新注册 pre-backward hook？**

因为原始前向时已经注册了 hook，重复注册会导致 hook 多次触发。

## 5. _pre_backward 是 AC 的"启动器"

```python
# _fsdp_state.py
def _pre_backward(self, module):
    # 设置 PRE_BACKWARD 状态
    # 这告诉后续的 _pre_forward（AC 重计算时）和 _post_forward：
    # "你现在在反向传播的上下文中"
    self._training_state = TrainingState.PRE_BACKWARD
    
    # 确保参数 unshard（为反向计算做准备）
    for fsdp_param_group in self._fsdp_param_groups:
        fsdp_param_group.pre_backward(default_prefetch=True)
        # pre_backward 内部：
        #   如果未 unshard → unshard() + wait_for_unshard()
        #   预取上一个模块的参数（反向预取）
```

### 5.1 Pre-Backward Hook 的注册时机

```python
# 在正常前向的 post_forward 中注册
# _fsdp_param_group.py
def pre_forward(self, module, args, kwargs):
    self._training_state = TrainingState.FORWARD
    self.unshard()
    self.wait_for_unshard()
    # 注册 _post_accumulate_grad_hook
    self._register_post_backward_hook(args, kwargs)
```

这个 hook 在**梯度累加完成后**触发，最终调用 `post_backward`。

## 6. AC + FSDP 的完整时序图

```text
Forward Pass:
═══════════════════════════════════════════════════════════════════

  model(x)
    │
    ├─ _pre_forward(model):      [TrainingState: IDLE → FORWARD]
    │    root_pre_forward:       stream sync + lazy init
    │    block0.pre_forward:     unshard block0, register hook
    │    ── 预取 block1 ──       (异步 unshard block1)
    │
    ├─ block0(x):                [使用完整参数计算]
    │
    ├─ _post_forward(block0):    [TrainingState: FORWARD → IDLE]
    │    block0.post_forward:    reshard block0
    │
    ├─ _pre_forward(block1):     [TrainingState: IDLE → FORWARD]
    │    block1.pre_forward:     wait (预取的 all-gather 可能已完成)
    │    ── 预取 block2 ──
    │    register hook ← 关键：注册 pre-backward hook
    │
    ├─ block1(x):                [AC: 不保存中间激活]
    │
    ├─ _post_forward(block1):    [TrainingState: FORWARD → IDLE]
    │    block1.post_forward:    reshard block1
    │
    ├─ _pre_forward(block2):
    │    ...
    ├─ block2(x):
    ├─ _post_forward(block2):
    │
    └─ 返回 output

Backward Pass:
═══════════════════════════════════════════════════════════════════

  loss.backward()
    │
    ├─ Grad 流入 block2:
    │    _pre_backward(block2):  [TrainingState: IDLE → PRE_BACKWARD]
    │         unshard block2
    │         反向预取 block1
    │    → 计算 block2 的梯度
    │    → post_backward(block2): reduce-scatter
    │
    ├─ Grad 流入 block1:
    │    _pre_backward(block1):  [TrainingState: IDLE → PRE_BACKWARD]
    │         unshard block1      ← 为 AC 重计算恢复参数
    │         反向预取 block0
    │
    │    === AC 重计算 block1 ===
    │    _pre_forward(block1):   [检测到 PRE_BACKWARD!]
    │         参数已 unshard → 跳过
    │    block1(x):              重计算前向（重新生成激活）
    │    _post_forward(block1):  [检测到 PRE_BACKWARD!]
    │         跳过 reshard
    │    === AC 重计算结束 ===
    │
    │    → 使用重计算的激活计算梯度
    │    → post_backward(block1): reduce-scatter
    │
    ├─ Grad 流入 block0:
    │    _pre_backward(block0):
    │         ...
    │    → 计算梯度 → reduce-scatter
    │
    └─ _root_post_backward_final_callback:
         确保所有 post_backward 完成
         清理所有状态 → IDLE
```

## 7. AC 的不同模式与 FSDP2 兼容性

### 7.1 `use_reentrant=False`（推荐）

```python
# PyTorch 1.11+ 的非重入 checkpoint
x = checkpoint(module, x, use_reentrant=False)

# 优势：
# - 不重入 autograd 图
# - 与 FSDP2 完美兼容
# - forward 被完整重新执行（包括 FSDP2 hooks）
```

### 7.2 `use_reentrant=True`（PyTorch < 1.11 默认）

```python
# 重入模式 - FSDP2 不支持！
x = checkpoint(module, x, use_reentrant=True)
# FSDP2 的 hook 机制与重入模式不兼容
```

### 7.3 Selective Activation Checkpointing (SAC)

```python
# SAC: 只对部分操作 checkpoint，而非整个模块
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

# 使用 checkpoint_wrapper 包装模块
wrapped_block = checkpoint_wrapper(
    block,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

# FSDP2 支持 module-hook-based AC
# 源码注释: "When composing with module-hook-based activation checkpointing,
#           the pre-backward hook is responsible for the unshard"
```

### 7.4 Auto SAC（自动选择 checkpoint 策略）

```python
# torch.distributed._tools.auto_sac
from torch.distributed._tools.auto_sac import get_auto_sac_policies, apply_auto_sac

# 基于内存预算自动决定哪些模块使用 AC
policies = get_auto_sac_policies(
    model,
    memory_budget=40 * 1024**3,  # 40 GB memory budget
)

apply_auto_sac(model, policies)
```

## 8. FSDP2 + AC 的包装顺序

### 正确顺序：先 FSDP，后 AC

```python
# ✅ 正确: FSDP 在内层，AC 在外层
for layer in model.layers:
    fully_shard(layer, mesh=mesh)     # 1. FSDP 包装
fully_shard(model, mesh=mesh)

# 然后对某些模块应用 AC
checkpoint(model.layer1, x)           # 2. AC 包装

# 原因：FSDP2 的 TrainingState 机制需要能在 AC 重计算时检测到 PRE_BACKWARD
# 如果 AC 在内层，FSDP2 的 hook 可能不会被正确触发
```

### 错误顺序：先 AC，后 FSDP

```python
# ❌ 错误: AC 在内层，FSDP 在外层
# AC 包装的模块在重计算时，FSDP2 的 hook 可能无法正确检测状态
```

## 9. 源码关键注释解读

```python
# _fsdp_state.py 中的关键注释：

# "When composing with module-hook-based activation checkpointing,
#  the pre-backward hook is responsible for the unshard"
#  含义：当使用基于 module hook 的 AC 时，
#  pre-backward hook 负责 unshard 参数（不是 pre-forward）

# "The PRE_BACKWARD state should be set before AC recomputation"
#  含义：PRE_BACKWARD 状态必须在 AC 重计算之前设置，
#  这样重计算时的 _pre_forward 才能正确检测并跳过

# "During PRE_BACKWARD, groups that are not currently unsharded
#  call unshard() and wait_for_unshard()"
#  含义：在 PRE_BACKWARD 状态中，未 unshard 的参数组会执行 unshard
```

## 10. 调试技巧

```python
# 检查 FSDP2 的 training state
import torch.distributed.fsdp._fully_shard._fsdp_state as fsdp_state

state = fsdp_state._get_module_fsdp_state(model.block1)
print(f"Training state: {state._training_state}")
# IDLE=0, FORWARD=1, PRE_BACKWARD=2, POST_BACKWARD=3

# 检查参数是否 unshard
for pg in state._fsdp_param_groups:
    print(f"Param group is_unsharded: {pg.is_unsharded}")
    print(f"Param group training state: {pg._training_state}")
```

---

> **源码参考**：
>
> - `TrainingState` 枚举和 AC 检测：`torch/distributed/fsdp/_fully_shard/_fsdp_state.py`
>   - `_pre_forward` 中的 `PRE_BACKWARD` 检测
>   - `_post_forward` 中的 `PRE_BACKWARD` short-circuit
> - `pre_backward` 设置 `PRE_BACKWARD` 状态：`_fsdp_param_group.py`
> - AC 兼容性注释：`_fsdp_state.py` 中的 `[Note: Activation Checkpointing]` 相关注释
> - Auto SAC: `torch/distributed/_tools/auto_sac.py`
