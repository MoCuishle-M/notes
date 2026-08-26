# EP 场景的 FSDP2 与重计算兼容性问题

> 基于 PyTorch 2.10.0 + torch_npu 2.10.0.post4 源码分析与实验验证
> 前置阅读：[FSDP2与重计算兼容性问题详解](./FSDP2与重计算兼容性问题详解.md)

## 0. 一句话结论

开启 EP（Expert Parallel）时，专家模块（experts）是一个独立的嵌套 FSDP 单元，和非 EP 场景下的 `linear_attn` 完全一样——重计算时如果走 `module.forward` 绕过 `__call__`，专家参数的 unshard hook 不会触发，同样报 "got mixed torch.Tensor and DTensor"。老方案（hook_module）在 EP 场景也需要打，就是把专家的 hook 上提。新方案（走 `__call__`）同样能解决 EP 场景，无需 hook_module。

## 1. EP 与 FSDP2 的关系

### 1.1 专家并行（Expert Parallel）

MoE（Mixture of Experts）模型有大量专家（Qwen3.5-35B-A3B 有 256 个专家），单卡放不下全部专家参数。EP 将专家参数沿专家维度切分到多卡：

```text
256 个专家，ep_size=2:
  rank 0: 持有专家 0-127 的参数
  rank 1: 持有专家 128-255 的参数
```

前向时通过 alltoall 将 token 路由到持有对应专家的卡上，计算完再 alltoall 回来。

### 1.2 专家模块是独立的 FSDP 单元

在 MindSpeed-MM 的 config（`qwen3_5_35B_config.yaml`）中，专家模块被单独 fully_shard：

```yaml
parallel:
  fsdp_plan:
    apply_modules:
      - model.language_model.layers.{*}              # decoder layer（父）
      - model.language_model.layers.{*}.linear_attn  # 注意力（子，独立 FSDP）
      - model.language_model.layers.{*}.mlp.experts  # 专家（子，独立 FSDP）
    hook_modules:
      - model.language_model.layers.{*}              # hook 上提到 layer 级
  expert_parallel_size: 1  # 非 EP 时为 1
  ep_plan:
    apply_modules:
      - model.language_model.layers.{*}.mlp.experts  # EP 切分对象
```

每个 `mlp.experts` 都被 `fully_shard` 单独包装，是一个独立的 FSDP 分片单元。它的参数（`gate_up_proj`、`down_proj`）被切分为 DTensor，前向时需要 all-gather 恢复。

### 1.3 EP 开启时的额外切分

当 `expert_parallel_size > 1` 时（`expert_fully_shard_parallel.py`），专家参数先沿 EP 维切片，再沿 FSDP 维分片：

```text
expert.gate_up_proj [256, hidden, intermediate]
  → EP 切片: [128, hidden, intermediate]（每卡 128 个专家）
  → FSDP 分片: DTensor Shard(1)（沿 hidden 维分片）
```

专家模块在 `ep_fsdp_mesh`（去掉 EP 维的 mesh）上被 `fully_shard` 包装，和普通 FSDP 单元一样通过 forward pre-hook 做 unshard。

## 2. EP 场景下的重计算问题

### 2.1 专家在重计算区内部

重计算（recompute）作用于 decoder layer 级别：

```yaml
recompute_plan:
  apply_modules:
    - model.language_model.layers.{*}
```

一个 decoder layer 的前向包含：`linear_attn` → `mlp.experts` → ...。这些子模块都在重计算区**内部**。

### 2.2 根因与非 EP 场景完全相同

重计算时，`checkpoint` 调用的是 `module.forward`（MindSpeed-MM 旧路径），绕过 `__call__`：

```text
反向重计算:
  checkpoint 调用 原 module.forward(hidden_states)
    → Qwen3_5MoeDecoderLayer.forward(hidden_states)
        → self.mlp(hidden_states)
            → self.experts(hidden_states)
                → expert.gate_up_proj(hidden_states)  # ❌ gate_up_proj.weight 是 DTensor
                                                      #     hidden_states 是普通 Tensor
                                                      #     → matmul 报错!
```

专家模块的 `fully_shard` pre-hook（unshard）挂在 `__call__` 上，重计算走 `module.forward` 绕过 `__call__`，hook 不触发，专家参数保持 DTensor → 报错。

**这和 `linear_attn` 的 `in_proj_qkv` 报错是同一个机制**——都是嵌套 FSDP 子单元的 hook 在重计算时不触发。EP 只是让专家模块成为了一个额外的嵌套 FSDP 单元。

### 2.3 实验 2 验证：VeOmni + EP 不报错

| 实验 | 框架 | ep_size | hook_module | 重计算路径 | 结果 |
|---|---|---|---|---|---|
| 2 | VeOmni | 2 | 无 | `__call__` | ✅ 20步跑通，loss 10.24→6.81 |

VeOmni 开 EP（`ep_size=2`，256 专家分到 2 卡各 128 个），重计算走 `__call__`，专家模块的 pre-hook 正常触发，不报错。这证明 EP 场景的根因同样是重计算的调用方式，不是 EP 本身的问题。

## 3. 老方案为何也需要 hook_module

### 3.1 expert_fully_shard_parallel.py 中的 use_hook_module

PR 2976 在 `expert_fully_shard_parallel.py` 中给 2.10.0 加了 `use_hook_module` 判定：

```python
use_hook_module = (
    "2.7.1" in torch.__version__
    or "2.9.0" in torch.__version__
    or "2.10.0" in torch.__version__  # PR 2976 新增
)
efsdp_hook_modules = get_fsdp_hook_modules(model, fsdp_plan) if use_hook_module else []

for experts in efsdp_modules:
    if use_hook_module:
        hook_module = find_hook_module(experts, efsdp_hook_modules)
        fsdp_kwargs = {**config, 'hook_module': hook_module}
    else:
        fsdp_kwargs = config
    fully_shard(experts, **fsdp_kwargs)
```

### 3.2 hook_module 在 EP 中的作用

和非 EP 场景完全一样：把专家模块的 forward pre/post hook **上提注册到父 decoder layer**（config 的 `hook_modules: model.language_model.layers.{*}`）。

这样在 decoder layer 的 `__call__` 边界（前向和反向 pre-backward），父 layer 的 hook 会统一 unshard 所有嵌套参数——包括 `linear_attn` 和 `mlp.experts`。即使重计算走 `module.forward` 绕过了子模块的 `__call__`，参数也已在父 layer 边界被 unshard。

### 3.3 为什么 EP 比 non-EP 更需要 hook_module

EP 场景下专家参数经历了双重切分（EP + FSDP），DTensor 的 placement 更复杂。重计算时如果不 unshard，不仅是 DTensor/Tensor 混用的问题，还可能导致 EP 维度的 alltoall 与 FSDP 维度的 all-gather 时序错乱。hook_module 通过在父 layer 边界统一管理所有嵌套单元的 unshard，避免了这种时序问题。

## 4. 新方案如何解决 EP 场景

### 4.1 走 __call__ 后专家的 hook 也正常触发

对 torch 2.10.0 的修改（`recompute.py` 的 `_enable_call_based_recompute`）：

```python
module.gradient_checkpointing = True
module._gradient_checkpointing_func = functools.partial(
    checkpoint, use_reentrant=use_reentrant, **ckpt_kwargs,
)
```

重计算时的调用链：

```text
反向重计算:
  checkpoint 调用 partial(super().__call__, **kwargs)
    → nn.Module.__call__(self, hidden_states)           # ✅ 走 __call__
        → _call_impl(hidden_states)
            ① forward pre-hooks
                → decoder_layer 的 FSDP pre-hook（unshard layer 参数）
                → experts 的 FSDP pre-hook（unshard 专家参数）  # ✅ 触发！
            ② self.forward(hidden_states)
                → self.mlp(hidden_states)
                    → self.experts(hidden_states)
                        → expert.gate_up_proj(hidden_states)
                            # gate_up_proj.weight 已 unshard → 完整 Tensor
                            # hidden_states 是 Tensor
                            # ✅ 不报错！
            ③ forward hooks（reshard）
```

走 `__call__` 后，**所有嵌套 FSDP 单元**（包括专家模块）的 pre-hook 都在重计算时正常触发，参数被 unshard。EP 场景和非 EP 场景同理解决。

### 4.2 expert_fully_shard_parallel.py 的改动

对 torch 2.10.0，`use_hook_module` 排除 2.10.0：

```python
use_hook_module = (
    "2.7.1" in torch.__version__
    or "2.9.0" in torch.__version__
    # 2.10.0 不需要：recompute 走 __call__，专家的 hook 正常触发
)
```

2.10.0 时 `use_hook_module=False`，`fully_shard(experts, **config)` 不传 `hook_module`，走标准 FSDP2。配合 recompute 走 `__call__`，专家参数在重计算时被正常 unshard。

## 5. 总结

| | 非 EP | EP |
|---|---|---|
| 嵌套 FSDP 单元 | `linear_attn` | `linear_attn` + `mlp.experts` |
| 根因 | 重计算走 `module.forward` 绕过 `__call__`，hook 不触发 | 完全相同 |
| 老方案（hook_module） | 把子单元 hook 上提到父 layer | 把子单元 hook 上提到父 layer（含专家） |
| 新方案（2.10.0） | 走 `__call__`，hook 正常触发 | 走 `__call__`，专家 hook 也正常触发 |

EP 场景和非 EP 场景的根因、修复方案完全一致——都是重计算调用方式的问题，与 EP 本身无关。
