# FSDP2 与重计算（Activation Checkpointing）兼容性问题详解

> 基于 PyTorch 2.10.0 + torch_npu 2.10.0.post4 源码分析与实验验证
> 实验环境：Ascend 910 (A3)，Qwen3.5-35B-A3B 模型，2 卡

## 0. 一句话结论

**PyTorch 2.10.0 本身不存在 FSDP2 与重计算的兼容性缺陷。** "got mixed torch.Tensor and DTensor" 报错的根因是：MindSpeed-MM 的重计算实现把 `module.forward` 传给 `torch.utils.checkpoint.checkpoint`，重计算时直接调 `module.forward(*args)`，**绕过了 `module.__call__`**，导致 FSDP2 挂在 `__call__` 上的 forward pre-hook（unshard）不触发。VeOmni 走 HuggingFace 标准路径，传 `partial(super().__call__, **kwargs)` 给 checkpoint，重计算时走 `__call__`，hook 正常触发，不需要任何 patch。

## 1. 背景

### 1.1 问题表现

MindSpeed-MM 在 PyTorch 2.10.0 上训练 Qwen3.5-35B（开启重计算 + FSDP2）时，反向传播的重计算阶段报错：

```
RuntimeError: aten.matmul.default: got mixed torch.Tensor and DTensor,
need to convert all torch.Tensor to DTensor before calling distributed operators!
```

报错位置在 `Qwen3_5MoeDecoderLayer.forward` 的 `self.in_proj_qkv(hidden_states)`——重计算时 `in_proj_qkv.weight` 是 DTensor（分片状态），而 `hidden_states` 是普通 Tensor。

### 1.2 两个框架的差异

| | MindSpeed-MM | VeOmni |
|---|---|---|
| torch 版本 | 2.10.0 | 2.10.0 |
| 模型 | Qwen3.5-35B-A3B | Qwen3.5-35B-A3B |
| FSDP2 | 自定义 fully_shard + hook_module | torch 原生 fully_shard |
| 重计算 | 替换 `module.forward`，传 `module.forward` 给 checkpoint | HF `gradient_checkpointing_enable`，传 `super().__call__` 给 checkpoint |
| hook_module patch | 需要（2.7.1/2.9.0/2.10.0） | 不需要 |
| 是否报错 | 不打 patch 时报错 | 不报错 |

同一个 torch 版本、同一个模型，两个框架表现不同——问题出在重计算的调用方式，不在 torch 本身。

### 1.3 PR 2976 的说法

MindSpeed-MM PR 2976（"feat: Add compatibility for PyTorch 2.10.0"）描述：

> "2.10.0 已在上游修复该问题。"

这句话指的是：torch 原生 FSDP2+AC 在**走 `__call__`** 时本就兼容（2.9.0 已修，2.10.0 沿用）。PR 之所以"还需要 patch"，是因为 MindSpeed-MM 自己的重计算实现绕过了 `__call__`——这是自找的场景，不是 torch 的缺陷。

## 2. 基础知识：重计算（Activation Checkpointing）

### 2.1 什么是重计算

重计算（又称 gradient checkpointing 或 activation checkpointing，简称 AC）通过**不保存前向中间激活值，反向时重新计算**来节省显存：

```text
无 AC:
  前向: 计算并保存所有中间激活 → 显存高
  反向: 使用保存的激活计算梯度

有 AC:
  前向: 计算但不保存 Checkpoint 区域内的激活 → 显存低
  反向: 重新执行 Checkpoint 区域的前向 → 重新生成激活 → 计算梯度
```

### 2.2 torch.utils.checkpoint.checkpoint 的工作原理

`torch.utils.checkpoint.checkpoint(function, *args, use_reentrant=False)` 的核心逻辑：

1. **前向**：在 `no_grad` 上下文中执行 `function(*args)`，只保存输入张量（`args`），不保存中间激活。
2. **反向**：重新执行 `function(*args)`（这就是"重计算"），这次在需要梯度的上下文中运行，重新生成中间激活，用于梯度计算。

关键点：**`function` 是什么，重计算时就调用什么。** 如果 `function` 是 `module.forward`，重计算时调 `module.forward(*args)`；如果 `function` 是 `module.__call__`，重计算时调 `module.__call__(*args)`。

### 2.3 use_reentrant=False（非重入模式）

PyTorch 2.10.0 推荐使用 `use_reentrant=False`（非重入模式）：

- 不重入 autograd 图，与 FSDP2 兼容
- 重计算时完整重新执行 `function`（包括其中的 hooks）
- MindSpeed-MM 和 VeOmni **都默认使用 `use_reentrant=False`**

所以问题不在 reentrant/non-reentrant 的选择上——两者都用 non-reentrant。差异在于传给 `checkpoint` 的 `function` 是什么。

## 3. 核心问题：module.forward vs \_\_call\_\_

### 3.1 `__call__` 是什么

`nn.Module.__call__` 是 Python 的魔术方法，当我们写 `output = module(x)` 时，Python 实际调用的是 `module.__call__(x)`。

`nn.Module.__call__` 的实现（`torch/nn/modules/module.py`）：

```python
def __call__(self, *args, **kwargs):
    return self._call_impl(*args, **kwargs)

def _call_impl(self, *args, **kwargs):
    # 1. 执行 forward pre-hooks（FSDP2 的 unshard 挂在这里！）
    for hook in self._forward_pre_hooks.values():
        result = hook(self, args, kwargs)  # FSDP2: all-gather 参数
    
    # 2. 执行 forward
    result = forward_call(*args, **kwargs)  # 即 self.forward(*args, **kwargs)
    
    # 3. 执行 forward hooks（FSDP2 的 reshard 挂在这里）
    for hook in self._forward_hooks.values():
        hook_result = hook(self, args, result)  # FSDP2: free 参数
    
    return result
```

**`__call__` 比 `forward` 多了两步：执行 pre-hooks 和 post-hooks。**

### 3.2 `module.forward` 是什么

`module.forward` 是模块定义的前向计算方法。直接调 `module.forward(x)` 只执行前向计算本身，**不经过 `_call_impl`，不触发任何 hook**。

### 3.3 两者的区别

```text
module(x)       →  __call__(x)  →  _call_impl(x):
                                     ① forward pre-hooks（FSDP2 unshard）
                                     ② self.forward(x)（实际计算）
                                     ③ forward hooks（FSDP2 reshard）

module.forward(x) → self.forward(x)（直接计算，跳过 ① 和 ③）
```

**FSDP2 的 unshard（all-gather 分片参数）是通过 `register_forward_pre_hook` 挂在 `_call_impl` 阶段的。** 只有走 `__call__` 才能触发；直接调 `module.forward` 会绕过它。

### 3.4 FSDP2 的 hook 挂载

FSDP2（`torch.distributed.fsdp._fully_shard`）通过 `register_forward_pre_hook` 注册 unshard 钩子（torch 源码文档原文：*"forward pre-hook <register_forward_pre_hook>"*）。当 `fully_shard(module)` 后：

- **前向时**：`module(x)` → `__call__` → `_call_impl` → pre-hook 触发 → all-gather 将分片参数恢复为完整参数 → `self.forward(x)` 用完整参数计算 → post-hook → reshard（释放完整参数，只保留分片）
- **反向重计算时**：需要再次 all-gather 恢复参数才能计算

如果重计算时走 `module.forward`，pre-hook 不触发，参数保持分片状态（DTensor），而输入是普通 Tensor → 混用报错。

## 4. MindSpeed-MM 的重计算路径（有问题）

### 4.1 recompute_modules 替换 module.forward

文件：`mindspeed_mm/fsdp/features/memory/recompute.py`

```python
def recompute_modules(model, plan, op_cache=None):
    ...
    for name, module in modules:
        module.forward = recompute_wrapper(module.forward, plan.use_reentrant, context_fn)
    return model
```

这里把 `module.forward` 替换为 `recompute_wrapper` 包装的版本。`recompute_wrapper` 内部：

```python
def recompute_wrapper(function, use_reentrant, context_fn=None):
    def wrapper(*args, **kwargs):
        ...
        return checkpoint(function, *args, use_reentrant=use_reentrant, **ckpt_kwargs, **kwargs)
    return wrapper
```

`function` 是被替换前的原始 `module.forward`。传给 `checkpoint` 的是 `module.forward`。

### 4.2 调用链分析

MindSpeed-MM 的 decoder layer 继承自 `GradientCheckpointingLayer`，但训练时从不调用 `gradient_checkpointing_enable()`，所以 `self.gradient_checkpointing` 恒为 `False`。

**前向传播：**
```
decoder_layer(x)                                    # 用户代码：调 __call__
  → GradientCheckpointingLayer.__call__(x)          # gradient_checkpointing=False
      → super().__call__(x)                          # 即 nn.Module.__call__
          → _call_impl(x)
              ① forward pre-hooks（FSDP2 unshard）   # ✅ 触发，参数恢复
              ② self.forward(x)                      # 已被替换成 recompute_wrapper
                  → checkpoint(原module.forward, x)
                      # no_grad 下执行原module.forward(x)，只保存输入 x
              ③ forward hooks（FSDP2 reshard）        # ✅ 触发，参数释放
```

前向时走 `__call__`，hook 正常触发，参数被 unshard → 不报错。

**反向重计算：**
```
torch.autograd 反向 → checkpoint 的 recomputation
  → 直接调用 function = 原 module.forward(x)          # ❌ 绕过 __call__！
      → Qwen3_5MoeDecoderLayer.forward(x)             # 原始前向逻辑
          → self.in_proj_qkv(hidden_states)           # weight 是 DTensor，hidden_states 是 Tensor
              → matmul 报错: "got mixed torch.Tensor and DTensor"
```

重计算时 `checkpoint` 直接调 `原module.forward(x)`，**绕过 `__call__`**。FSDP2 的 pre-hook 不触发，参数保持分片状态（DTensor），而输入是普通 Tensor → matmul 报错。

### 4.3 hook_module 方案（MindSpeed-MM 的 workaround）

为了解决上述问题，MindSpeed-MM 的 `hook_module` 机制：

1. 把重算区内部的嵌套 FSDP 单元（`linear_attn`、`mlp.experts`）的 forward pre/post hook **上提注册到父 layer**（config 的 `hook_modules: model.language_model.layers.{*}`）
2. 在父 layer 的 `__call__` 边界统一 unshard 全部嵌套参数
3. 这样即使子模块的 `__call__` 被绕过，参数也已在父 layer 边界被 unshard

**hook_module 兜的是"重计算走 `module.forward` 绕过 `__call__`"这个自找的场景。**

## 5. VeOmni 的重计算路径（无问题）

### 5.1 使用 HF gradient_checkpointing_enable

文件：`veomni/distributed/torch_parallelize.py`

```python
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False, ...},
)
```

VeOmni 使用 HuggingFace 标准 API 启用重计算，不替换 `module.forward`。

### 5.2 GradientCheckpointingLayer.\_\_call\_\_ 的处理

HF `GradientCheckpointingLayer.__call__`（transformers 5.2.0，`modeling_layers.py`）：

```python
class GradientCheckpointingLayer(nn.Module):
    gradient_checkpointing = False

    def __call__(self, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            # 处理 past_key_values 等（设为 None）
            ...
            return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
        return super().__call__(*args, **kwargs)
```

关键：传给 `checkpoint` 的是 `partial(super().__call__, **kwargs)`，即 `nn.Module.__call__` 绑定到 self。

HF docstring 原文：*"We pass the `__call__` method of the modules instead of `forward` because `__call__` attaches all the hooks of the module."*

### 5.3 调用链分析

**前向传播：**
```
decoder_layer(x)                                    # 用户代码：调 __call__
  → GradientCheckpointingLayer.__call__(x)          # gradient_checkpointing=True, training=True
      → self._gradient_checkpointing_func(          # 即 checkpoint(partial(super().__call__), x)
            partial(super().__call__, **kwargs), x)
          )
          → checkpoint(nn.Module.__call__, x, use_reentrant=False)
              # no_grad 下执行 nn.Module.__call__(x)
              → _call_impl(x)
                  ① forward pre-hooks（FSDP2 unshard）  # ✅ 触发
                  ② self.forward(x)                      # 原始 forward，未被替换
                  ③ forward hooks（FSDP2 reshard）        # ✅ 触发
```

**反向重计算：**
```
torch.autograd 反向 → checkpoint 的 recomputation
  → 调用 function = partial(super().__call__, **kwargs)
      → nn.Module.__call__(self, x)                   # ✅ 走 __call__！
          → _call_impl(x)
              ① forward pre-hooks（FSDP2 unshard）     # ✅ 触发！参数恢复
              ② self.forward(x)                        # 原始 forward，用完整参数计算
              ③ forward hooks（FSDP2 reshard）          # ✅ 触发
```

重计算时走 `__call__`，FSDP2 的 pre-hook 正常触发，参数被 unshard → **不报错**。

### 5.4 为什么不会递归

有人可能担心：`__call__` 里调 `checkpoint(super().__call__)`，重计算时又调 `super().__call__` → `self.forward()` → 不会再进 `__call__` 吗？

不会。因为 `super().__call__` 是 `nn.Module.__call__`（不是 `GradientCheckpointingLayer.__call__`）。`nn.Module.__call__` → `_call_impl` → `self.forward()` 是原始 forward，不经过 `GradientCheckpointingLayer.__call__` 的 checkpoint 分支。所以不会递归。

## 6. 两种路径的对比总结

| | MindSpeed-MM（旧） | VeOmni / MindSpeed-MM（新，2.10.0） |
|---|---|---|
| 启用方式 | 替换 `module.forward` | `gradient_checkpointing_enable` |
| 传给 checkpoint 的 function | `module.forward`（绕过 `__call__`） | `partial(super().__call__)`（走 `__call__`） |
| 重计算时 hook 触发 | ❌ 不触发 | ✅ 触发 |
| 重计算时参数状态 | DTensor（分片） | 完整 Tensor（unshard） |
| 是否报错 | ❌ 报 mixed Tensor/DTensor | ✅ 不报错 |
| 需要 hook_module | 是 | 否 |

## 7. 实验验证

### 7.1 三组实验

| 实验 | 框架 | EP | hook_module | 重计算路径 | 结果 |
|---|---|---|---|---|---|
| 1 | VeOmni | 关 | 无此机制 | `__call__` | ✅ 20步跑通，loss 10.24→6.81 |
| 2 | VeOmni | ep_size=2 | 无此机制 | `__call__` | ✅ 20步跑通，loss 10.24→6.81 |
| 3 | MindSpeed-MM | 关 | 强制关闭 | `module.forward` | ❌ 第1步报 mixed Tensor/DTensor |

### 7.2 实验 3 报错

```
RuntimeError: aten.matmul.default: got mixed torch.Tensor and DTensor,
need to convert all torch.Tensor to DTensor before calling distributed operators!
```

报错位置：`modeling_qwen3_5_moe.py:696` 的 `self.in_proj_qkv(hidden_states)`。

### 7.3 结论

- torch 2.10.0 本身没有 FSDP2+AC 兼容性缺陷
- 报错根因是 MindSpeed-MM 传 `module.forward` 给 checkpoint，绕过 `__call__`
- VeOmni 传 `super().__call__` 给 checkpoint，走 `__call__`，不报错
- hook_module 是针对"绕过 `__call__`"这个自找场景的兜底 patch
- 对于 torch 2.10.0，修改重计算调用方式走 `__call__` 即可不需要 hook_module

## 8. torch 2.10.0 的修复方案

对 torch 2.10.0，MindSpeed-MM 的修改（仅 2.10.0，不影响 2.7.1/2.9.0）：

1. **`recompute.py`**：对 2.10.0，不再替换 `module.forward`，改为设置 `gradient_checkpointing=True` + `_gradient_checkpointing_func`，走 `GradientCheckpointingLayer.__call__` 路径
2. **`fully_shard_parallel.py`**：`use_hook_module` 排除 2.10.0
3. **`expert_fully_shard_parallel.py`**：同上

详见 git diff。

## 附：相关源码位置

| 文件 | 说明 |
|---|---|
| `mindspeed_mm/fsdp/features/memory/recompute.py` | MindSpeed-MM 重计算实现 |
| `mindspeed_mm/fsdp/distributed/fully_shard_parallel.py` | use_hook_module 判定 |
| `mindspeed_mm/fsdp/distributed/expert_parallel/expert_fully_shard_parallel.py` | EP use_hook_module 判定 |
| `mindspeed_mm/fsdp/ops/fully_shard/fully_shard.py` | hook_module_init / apply_fully_shard_patch |
| `transformers/modeling_layers.py` | GradientCheckpointingLayer.__call__ |
| `torch/nn/modules/module.py` | nn.Module.__call__ / _call_impl |
| `torch/distributed/fsdp/_fully_shard/_fully_shard.py` | FSDP2 register_forward_pre_hook |
