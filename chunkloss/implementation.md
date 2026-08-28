# MindSpeed-MM ChunkLoss 源码实现

> 源码基线：`/home/zhangxubin/workspace/code/MindSpeed-MM`，提交 `192b2a13`。本文中的路径均相对于该仓库根目录。

## 1. 总体调用链

原生 FSDP2 的主要链路如下：

```text
YAML features
  │
  ├─ FeatureArguments / ChunkLossPlanConfig
  │      参数校验与 legacy/cce 归一化
  │
  ├─ FeaturesApplier.apply_chunkloss()
  │      给模型设置开关，并替换目标 lm_head.forward
  │
  ├─ TrainEngine.set_loss_func(batch_data)
  │      每个 batch 构造 loss_function 闭包
  │
  └─ 模型 forward
         hidden_states ── patched lm_head(hidden_states, loss_function)
                              │
                              ├─ legacy: ChunkLoss.apply
                              └─ cce:    ChunkLossCceFused.apply
```

普通路径是 `hidden_states -> lm_head -> logits -> loss_function(logits)`；ChunkLoss 路径把 loss 计算前移到 lm_head 内部，使 loss 函数同时拿到 hidden states 和 lm_head 权重，从源头阻止完整 logits 返回到模型顶层。

## 2. 配置模型与参数归一化

文件：`mindspeed_mm/fsdp/params/feature_args.py`

### 2.1 参数定义

`ChunkLossPlanConfig` 定义：

- `apply_module="lm_head"`：要替换的模块名匹配表达式；
- `chunk_size=None`：legacy 下缺省会规范化为 1024，CCE 下保持 `None`；
- `total_chunk_size=4096`：动态分块总 token 上限；
- `impl_type="legacy"`：可选 `legacy` 或 `cce`；
- `vocab_tile_size=4096`：CCE 的词表 tile 大小。

`FeatureArguments` 另有两个开关：`enable_chunk_loss` 和 `enable_dynamic_chunk_loss`。

### 2.2 `_normalize_loss_plan`

模型校验器是分支选择的重要前置条件：

```text
未开启 enable_chunk_loss
  └─ plan.chunk_size = None
  └─ plan.vocab_tile_size = None

开启 + legacy
  └─ chunk_size 缺省 -> 1024
  └─ vocab_tile_size -> None

开启 + cce
  └─ 保留 vocab_tile_size
  └─ chunk_size 缺省时仍为 None
```

CCE 必须显式开启 `enable_chunk_loss`，否则校验直接报错。这能避免用户只写 `impl_type: cce`，实际上却静默走普通 loss。

注意：动态模式由 `enable_dynamic_chunk_loss` 驱动，训练引擎会额外传入 `total_chunk_size`；它属于 legacy 分块逻辑，不能与 CCE 组合。

## 3. 在 FSDP 分片前替换 lm_head

文件：

- `mindspeed_mm/fsdp/features/apply_features.py`
- `mindspeed_mm/fsdp/features/memory/chunkloss/chunkloss_lm_head.py`

`FeaturesApplier.pre_fully_shard_apply()` 在执行 FSDP fully-shard 之前调用 `apply_chunkloss()`：

1. 静态模式给模型设置 `enable_chunk_loss=True` 和 `chunk_size`；
2. 动态模式设置 `enable_dynamic_chunk_loss=True`；
3. `get_chunkloss_module()` 遍历 `named_modules()`，按 `apply_module` 匹配模块；
4. 匹配数必须恰好为 1，目标必须是 `torch.nn.Linear`；
5. `types.MethodType` 把目标模块实例的 `forward` 替换为 `chunkloss_forward`。

替换后的接口从：

```python
lm_head(hidden_states) -> logits
```

变成：

```python
lm_head(hidden_states, loss_func, labels=None) -> scalar_loss
```

如果 FSDP2 已把 weight 表示为 `DTensor`，新 forward 会先通过 `to_local()` 取得当前 rank 的本地权重；bias 必须与 weight 同为 DTensor，或两者都不是 DTensor。随后调用：

```python
loss_func(hidden_states, local_weight, local_bias, labels)
```

这个替换发生在 fully-shard 前，有利于后续模型 forward 始终通过同一 lm_head 实例进入定制逻辑。

## 4. 每个 batch 构造 loss 闭包

文件：`mindspeed_mm/fsdp/train/train_engine.py`

`TrainEngine.set_loss_func(batch_data)` 是配置与实际计算的汇合点：

```python
plan = args.features.chunkloss_plan
chunk_size = plan.chunk_size
vocab_tile_size = plan.vocab_tile_size

if args.features.enable_dynamic_chunk_loss:
    batch_data["total_chunk_size"] = plan.total_chunk_size

loss_func = build_loss_func(
    args.features.loss_cfg.loss_type,
    chunk_size=chunk_size,
    vocab_tile_size=vocab_tile_size,
    **batch_data,
)
self.model.loss_function = loss_func
```

之所以每个 batch 重新构造，是因为 labels、data packing 的 `cu_seqlens`、每 step 平均 token 数，以及动态 batch size 都可能变化。

当 `loss_type == "raw"` 时该方法直接返回，不注入预置 loss。这也是原生 FSDP2 下 ChunkLoss 不能与 raw model loss 混用的原因。

## 5. 模型 forward 如何绕过完整 logits

以 `mindspeed_mm/fsdp/models/qwen3_5/modeling_qwen3_5.py` 为例：

```python
if self.enable_chunk_loss or self.enable_dynamic_chunk_loss:
    logits = None
    loss = self.lm_head(hidden_states, self.loss_function)
else:
    logits = self.lm_head(hidden_states)
    loss = self.loss_function(logits=logits, labels=labels, ...)
```

这段分支很关键：只有不执行普通 `self.lm_head(hidden_states)`，才能真正避免完整 logits。不同模型的适配代码位置不同，但原则相同：模型主体返回 hidden states，顶层 lm_head 在 ChunkLoss 分支中直接返回 loss。

当前仓库中 Qwen3.5、Qwen3.5-MoE、Qwen3-Omni、MiniMax-M3-VL、MTP 等原生 FSDP2 模型都能看到类似开关。Kimi-K2.5 和部分模型存在更早的模型内自建 loss 闭包逻辑，阅读时要区分“训练引擎统一注入”和“模型自身构造”两种适配方式。

## 6. `build_loss_func` 的三条分支

文件：`mindspeed_mm/fsdp/loss/loss_func.py`

`build_loss_func()` 根据规范化后的参数返回不同签名的闭包：

| 条件 | 返回闭包输入 | 实现路径 |
|---|---|---|
| `vocab_tile_size is not None` | hidden、weight、bias、labels | CCE |
| `chunk_size` 或 `total_chunk_size` 有效 | hidden、weight、bias、labels | legacy ChunkLoss |
| 两者都无效 | logits、labels | 普通 cross entropy |

这个设计让模型 forward 可以通过“是否启用 ChunkLoss”决定调用签名，而闭包内部再决定具体实现。

### 6.1 label 与归一化参数准备

`get_loss_func_params()` 负责所有路径共用的语义：

1. 动态模式调用 `calculate_chunk_size(B, total_chunk_size)`；
2. labels 右侧 pad `-100` 后左移一位，形成 next-token `shift_labels`；
3. `loss_mask = shift_labels > -1`；
4. 根据 `loss_type` 计算 `alpha` 和 reduction；
5. context parallel 开启时，对 labels 和必要的 alpha 做对应切分；
6. legacy 模式按序列维切 labels，生成与 hidden chunk 一一对应的 kwargs 列表。

`fixed_cross_entropy()` 最终只支持 `sum` 和 `none` 两种底层 reduction，再结合 `alpha` 实现三类上层 loss 语义。

## 7. legacy 自定义 autograd

文件：`mindspeed_mm/fsdp/features/memory/chunkloss/chunkloss.py`

### 7.1 `calculate_lm_loss`

单块计算过程是：

```python
hidden = hidden.reshape(-1, H)
logits = F.linear(hidden, head_weight).float()
loss = fixed_cross_entropy(logits, shift_labels.reshape(-1), ...)
```

lm_head bias 没有参与计算。logits 被显式转为 FP32 后进入交叉熵，这是数值稳定性更好但显存更高的原因之一，也说明减小 chunk 对峰值很敏感。

### 7.2 `ChunkLoss.forward`

实现先分配：

```python
grad_inputs = torch.empty_like(hidden_states)
grad_weight = torch.zeros_like(head_weight)
```

再把 hidden states 和 `grad_inputs` 都沿 `dim=1` 按 `chunk_size` 切分。对每块执行：

```python
(d_hidden_i, d_weight_i), (loss_i, _) = torch.func.grad_and_value(
    loss_forward, argnums=(0, 1), has_aux=True
)(hidden_i, head_weight, None, **chunk_kwargs_i)
```

然后：

```python
accumulated_loss += loss_i
grad_inputs_i.copy_(d_hidden_i)
grad_weight += d_weight_i
```

循环结束后，只把 `grad_inputs` 和 `grad_weight` 存入 autograd context。

### 7.3 `ChunkLoss.backward`

backward 取出预计算梯度。如果上游梯度不是 1，就同时缩放 hidden 和 weight 梯度，最后返回：

```text
(grad_input, grad_weight, None, None, None, None)
```

因此 bias、Python callable、kwargs 列表和 chunk size 都没有梯度。

这种设计本质上是“forward 中完成局部 forward+backward，外层 backward 负责接回全局 autograd 图”。它避免保存每个 logits chunk 的计算图直到训练总 loss backward。

## 8. CCE 自定义 autograd

文件：

- `mindspeed_mm/fsdp/features/memory/chunkloss/chunkloss_cce_fused.py`
- `mindspeed_mm/fsdp/features/memory/chunkloss/chunkloss_cce_kernels.py`

### 8.1 公共入口

`chunk_loss_cce_fused()` 将 $[B,S,H]$ 和 $[B,S]$ 展平为 $[N,H]$、$[N]$。如果配置了有效 `seq_chunk_size`，它再按展平 token 范围调用多个 `ChunkLossCceFused.apply()` 并累加 loss；否则单次处理全部 token。

Triton kernel 在函数内部延迟导入，隔离了可选依赖。

### 8.2 online log-sum-exp

设当前累计状态为最大值 $m$ 与移位指数和 $s$，新词表 tile 的状态为 $m_t$、$s_t$，合并公式为：

$$
m_{\text{new}}=\max(m,m_t),
\qquad
s_{\text{new}}
=s\,e^{m-m_{\text{new}}}
+s_t\,e^{m_t-m_{\text{new}}},
\qquad
\mathrm{LSE}=m_{\text{new}}+\log s_{\text{new}}
$$

同时检查 label 是否落在当前词表 tile，若命中则抽取正确类别 logit。这样只需保存每个 token 的 $m$、$s$、正确类 logit，最终得到 $\mathrm{LSE}-\mathrm{logit}_{y}$，不需要完整词表 logits。

### 8.3 多 stream 流水

代码创建并缓存两个设备 stream：

- `stream_cube`：矩阵乘，包括前向投影、反向重算、`dW` 和 `dH`；
- `stream_vec`：Triton online-softmax 与 `softmax-onehot` tile kernel。

3 个 $[N,\text{vocab\_tile\_size}]$ buffer 轮转使用。每个 tile 对应若干 event，约束“buffer 可复用、矩阵乘完成、向量 kernel 完成、梯度矩阵乘完成”的顺序，并预取下一个 tile 的重算矩阵乘。

### 8.4 CCE backward

前向保存的主要张量形状为：$\mathrm{hidden}\in\mathbb{R}^{N\times H}$、
$\mathrm{weight}\in\mathbb{R}^{V\times H}$、$\mathrm{labels}\in\mathbb{R}^{N}$、
$\mathrm{lse}\in\mathbb{R}^{N}$。

反向逐词表 tile：

1. 重算 $\mathrm{hidden}\,\mathrm{weight}_{\mathrm{tile}}^{\mathsf T}$；
2. 原地改写 tile 为 $\mathrm{softmax}-\mathrm{onehot}$，ignore token 置零；
3. $dW_{\mathrm{tile}}=G_{\mathrm{tile}}^{\mathsf T}\mathrm{hidden}$；
4. $dH \leftarrow dH+G_{\mathrm{tile}}\mathrm{weight}_{\mathrm{tile}}$。

最后按上游 `grad_output` 缩放并返回 hidden 与 weight 梯度。CCE loss 在 `build_loss_func()` 外层再除以 $\alpha$，autograd 会把这个归一化自动传入 `grad_output`。

## 9. Megatron-FSDP2 过渡路径

文件：

- `mindspeed_mm/models/transformers_model.py`
- `mindspeed_mm/models/common/chunkloss.py`

这一路径从 `model.json` 的 `loss_cfg.compute_mode` 读取：

- `default`：普通 logits loss；
- `chunk`：固定 `chunk_size`；
- `dynamic_chunk`：把配置的 `chunk_size` 当作总 token 上限，运行时计算真实块长。

`TransformersModel.build_loss_ctx()` 负责 label、mask、alpha 和闭包构造；模型实现需要接收 `loss_ctx`，并在内部 lm_head 位置调用它。核心 `ChunkLoss` 代码与原生 FSDP2 的 legacy 文件基本同构，但该路径还保留 `token_loss`、`square_loss` 等历史 loss 类型。

仓库入口文档已明确把该后端标为过渡态；新增模型应优先阅读和接入原生 FSDP2 统一链路。

## 10. 测试覆盖与当前缺口

相关测试：

- `tests/ut/loss/test_chunkloss.py`：在 NPU/BF16 大形状上比较非分块与分块的 loss、lm_head weight 梯度；覆盖 default、per-sample、per-token。
- `tests/ut_fsdp/loss/test_loss_func.py`：验证原生 FSDP2 的 chunk/non-chunk loss 数值一致、错误处理和 packing 语义。
- `tests/ut_fsdp/loss/test_dynamic_chunkloss.py`：验证动态块长公式、边界退化与 `build_loss_func` 分支选择。

测试中使用的主要容差为：loss（$\mathrm{rtol}=10^{-5}$、$\mathrm{atol}=10^{-6}$），梯度（$\mathrm{rtol}=10^{-4}$、$\mathrm{atol}=10^{-5}$）。这体现的是浮点容差内等价，而非逐 bit 相同。

在当前提交中，没有找到针对 CCE kernel 的仓库单元测试。CCE 是近期接入的 NPU/Triton 路径，使用前应在目标硬件上额外完成：

1. 与普通/legacy loss 的前向数值对比；
2. hidden 与 lm_head weight 梯度对比；
3. ignore token、尾部 vocab tile 和非整除 token segment 测试；
4. 多次迭代稳定性与 stream/event 正确性测试；
5. 峰值显存、吞吐和 loss 曲线对比。

## 11. 代码阅读时容易误解的点

### 11.1 注释中的 “feature dimension”

`ChunkLoss` 类 docstring 写过 “feature dimension”，但实际 `torch.split(..., dim=1)`；对 $[B,S,H]$ 输入，切的是序列维，不是 hidden feature 维。

### 11.2 “逐块反向”发生在自定义 forward 内

用户看到训练代码仍然只调用一次 `loss.backward()`，容易以为每块计算图都保留到了最后。实际局部梯度已经由 `torch.func.grad_and_value` 在 `ChunkLoss.forward` 中算出，外层 backward 只是返回预存梯度。

### 11.3 legacy 和 CCE 的 `chunk_size` 语义不同

- legacy：沿 $[B,S,H]$ 的 $S$ 切，每个块约 $B\times\text{chunk\_size}$ 个 token；
- CCE：公共入口先展平为 $N=B\times S$，可选外层分段按 $N$ 切。

比较参数或迁移配置时不能直接把两者视为同一单位。

### 11.4 仅替换 lm_head 还不够

模型 forward 必须在开关开启时调用 `lm_head(hidden_states, loss_function)`，而不是先执行普通 lm_head 得到 logits。否则即使注册了特性，也无法获得预期显存收益。

## 12. 一句话总结实现

MindSpeed-MM ChunkLoss 通过“把 loss 注入 lm_head、让 lm_head 直接消费 hidden states”绕过完整 logits；legacy 路径沿序列分块并在自定义 forward 中预计算梯度，CCE 路径再沿词表 tile 流式执行 online softmax 与梯度重算，从而用更多调度和重算换取更低的 loss 阶段显存峰值。
