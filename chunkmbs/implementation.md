# MindSpeed-MM ChunkMBS 核心代码实现

> 源码基线：`/home/zhangxubin/workspace/code/MindSpeed-MM`，提交 `192b2a139d0c`。本文路径均相对于该仓库根目录。

## 1. 总体调用链

```text
YAML: features.enable_chunk_mbs / chunkmbs_plan
  │
  ▼
FeatureArguments + ChunkMbsPlanConfig
  │
  ▼
Trainer.get_model()
  │
  ├─ FeaturesApplier.pre_fully_shard_apply(model)
  │    ├─ apply_recompute_models
  │    ├─ apply_activation_offload_modules
  │    └─ apply_chunk_mbs
  │          ├─ get_chunkmbs_modules
  │          └─ apply_chunkmbs_module
  │                 module.forward = chunk_mbs_forward(...)(old_forward)
  │
  ├─ model_parallel_applier(model)
  │    └─ fully_shard(target modules)
  │
  └─ 训练时 module(...)
       └─ FSDP unshard
            └─ ChunkMBS wrapper
                 ├─ slice inputs
                 ├─ old_forward(各个 chunk)
                 └─ cat outputs
       └─ FSDP reshard
```

核心实现集中在 `mindspeed_mm/fsdp/features/communication/chunk_mbs.py`，约 196 行。它没有专用算子或自定义 autograd，主要依靠 Python decorator、Tensor slice 和 `torch.cat`。

## 2. 配置数据结构

文件：`mindspeed_mm/fsdp/params/feature_args.py`

`ChunkMbsPlanConfig` 定义五个字段：

```python
class ChunkMbsPlanConfig(BaseArguments):
    apply_modules: List[str] = None
    chunk_mbs: int = 1
    batch_dim: int = 0
    chunk_arg_indexs: List[int] = [0]
    chunk_kwarg_names: List[str] = []
```

`FeatureArguments` 再定义总开关与 plan：

```python
enable_chunk_mbs: bool = False
chunkmbs_plan: ChunkMbsPlanConfig = field(default_factory=ChunkMbsPlanConfig)
```

当前这里仅提供类型和默认值，没有为 ChunkMBS 定义独立的模型校验器。因此 `chunk_mbs<=0`、错误索引、不同输入 batch 长度等问题会延迟到首次 forward 才暴露。

## 3. 特性安装时机与 wrapper 顺序

文件：`mindspeed_mm/fsdp/train/trainer.py`

`Trainer.get_model()` 先构建模型和 LoRA，再执行：

```python
self.model_features_applier.pre_fully_shard_apply(model)
model = self.model_parallel_applier(model)
self.model_features_applier.post_fully_shard_apply(model)
```

这说明 ChunkMBS 是在 `fully_shard` 之前安装到原始 module 实例上的。

文件：`mindspeed_mm/fsdp/features/apply_features.py`

`pre_fully_shard_apply()` 的调用顺序是：

```python
self.apply_recompute_models(model)
self.apply_activation_offload_modules(model)
self.apply_chunk_mbs(model)
self.apply_chunkloss(model)
```

每一步都是把当前 `module.forward` 捕获为 `old_forward`，再替换成新 wrapper。因此后安装的 ChunkMBS 在调用栈最外层：

```text
chunk_mbs_wrapper(
  activation_offload_wrapper(
    recompute_wrapper(
      original_forward
    )
  )
)
```

这个顺序直接决定语义：先把大输入切成 chunk，每个 chunk 再分别进入 offload 和 checkpoint。若把 ChunkMBS 安装在 recompute 内层，checkpoint 边界仍会面对完整 batch，无法得到相同的瞬时激活收益。

`model_parallel_applier` 随后在 `fully_shard_parallel.py` 对匹配的 module 调用 `fully_shard()`。FSDP 管理的是 module 的一次整体调用；ChunkMBS 的 Python 循环处在该调用内部，所以所有 chunk 共享这一轮 unsharded 参数。

## 4. 目标模块匹配

文件：`mindspeed_mm/fsdp/features/communication/chunk_mbs.py`

`get_chunkmbs_modules(model, plan)` 对每个 plan 表达式遍历 `model.named_modules()`，通过 MindSpeed 的 `module_name_match(plan_name, name)` 进行匹配：

```python
for plan_name in plan:
    for name, module in modules.named_modules():
        if module_name_match(plan_name, name):
            matched_modules.append((name, module))
```

若一个 module 都没有匹配，抛出：

```text
[ChunkMBS] No module named <plan>.
```

与 `FeaturesApplier.get_needed_modules()` 不同，这里没有 `(name, module) not in matched_modules` 的去重判断。因此重叠 plan 可能使同一 module 被多次加入并多层包装，应避免同时配置能命中同一层的表达式。

## 5. forward monkey patch

`apply_chunkmbs_module()` 遍历匹配结果，打印日志并替换实例的 forward：

```python
module.forward = chunk_mbs_forward(
    chunk_mbs=cfg.chunk_mbs,
    batch_dim=cfg.batch_dim,
    chunk_arg_indexs=cfg.chunk_arg_indexs,
    chunk_kwarg_names=cfg.chunk_kwarg_names,
)(module.forward)
```

`functools.wraps` 保留了被包装函数的名称和元信息。这里直接给 module 实例赋可调用对象，而不是修改 module 类；影响范围仅是已匹配的实例。

## 6. batch size 推断与快速路径

`chunk_mbs_forward()` 是 decorator factory，实际逻辑在内部 `wrapper(*args, **kwargs)`：

```python
if chunk_arg_indexs:
    full_batch_size = args[chunk_arg_indexs[0]].shape[batch_dim]
elif chunk_kwarg_names:
    full_batch_size = kwargs[chunk_kwarg_names[0]].shape[batch_dim]
else:
    raise ValueError("No tensor input found to infer batch size.")
```

位置参数拥有固定优先级。代码只用第一个配置项推断 `B`，不会检查其他被切 Tensor 是否也等长。

快速路径是：

```python
if full_batch_size <= chunk_mbs:
    return forward_func(*args, **kwargs)
```

因此验证或推理 batch 小于等于 chunk 时不会发生额外 slice/cat。但 wrapper 仍已存在，且在调用前已经完成 batch size 推断。

## 7. 递归输入切分

辅助函数 `_slice_batch_recursive(data, start, end, batch_dim)` 支持：

```text
Tensor       -> data[..., start:end, ...]
tuple/list   -> 保持容器类型，递归切每一项
dict         -> 保持 key，递归切每个 value
其他类型     -> 原样返回
```

Tensor 实现先构造与 `ndim` 等长的 slice 列表，只替换 `batch_dim`：

```python
slices = [slice(None)] * data.ndim
slices[batch_dim] = slice(start, end)
return data[tuple(slices)]
```

这通常返回 view，切分本身不复制整份数据，梯度也能回传到原 Tensor。

需要注意，配置粒度是“整个 arg/kwarg”。如果一个 dict 同时含 batch Tensor 和非 batch Tensor，只要该 dict 被列入切分，内部所有 Tensor 都会沿同一维切片；实现无法按嵌套路径排除单个字段。

## 8. chunk 循环

块数采用向上取整：

```python
num_micros = (full_batch_size + chunk_mbs - 1) // chunk_mbs
```

这段整数运算等价于数学公式

$$
K=\left\lceil\frac{B}{C}\right\rceil,
$$

其中 $B$ 对应 `full_batch_size`，$C$ 对应 `chunk_mbs`，$K$ 对应 `num_micros`。$\lceil\cdot\rceil$ 是向上取整，`//` 是正整数的向下整除；先加 $C-1$，可以把非整除时的余数“进一块”。例如 $B=5$、$C=2$ 时，代码计算 `(5+2-1)//2=3`，与 $\lceil5/2\rceil=3$ 一致。

第 `i` 块边界：

```python
start = i * chunk_mbs
end = min(start + chunk_mbs, full_batch_size)
```

$i$ 从 0 到 $K-1$。`start` 是当前块的起始下标（包含），`end` 是结束下标（不包含）；`min` 保证最后一块不会越过完整 batch 的末尾。继续使用 $B=5$、$C=2$ 的例子，三个半开区间依次为 $[0,2)$、$[2,4)$、$[4,5)$，对应 Python 切片 `0:2`、`2:4`、`4:5`，因此没有遗漏或重复样本。

随后分别构建位置参数和关键字参数：

```python
micro_args = [
    slice_recursive(arg) if arg_idx in chunk_arg_indexs else arg
    for arg_idx, arg in enumerate(args)
]

micro_kwargs = {
    name: slice_recursive(value) if name in chunk_kwarg_names else value
    for name, value in kwargs.items()
}
```

每块顺序调用捕获的 `forward_func`，结果存入 Python list：

```python
out = forward_func(*micro_args, **micro_kwargs)
outputs.append(out)
```

这里没有并行执行 chunk。性能收益来自减少 FSDP 参数通信和控制激活，而不是让 chunk 彼此并发。

## 9. 输出拼接与 autograd

输出为 Tensor 时：

```python
return torch.cat(outputs, dim=batch_dim)
```

输出为 tuple/list 时，代码按输出位置转置后分别拼接：

```python
return type(outputs[0])(
    torch.cat([out[i] for out in outputs], dim=batch_dim)
    for i in range(len(outputs[0]))
)
```

例如两个 chunk 都返回 `(hidden, aux)`，结果是：

```text
(cat(hidden_0, hidden_1), cat(aux_0, aux_1))
```

其他输出类型直接抛 `TypeError`。这里没有递归拼接，也不支持对 batch 无关的标量取第一个或聚合，所以模型输出适配是启用前的重要检查项。

`slice -> forward -> cat` 都是标准 PyTorch 可微操作。backward 时 `cat` 的梯度自然拆给每个 `out_i`，各 chunk 再回传到输入 slice 和同一组 module 参数；共享参数梯度由 autograd 累加。

## 10. recompute 的配合实现

文件：`mindspeed_mm/fsdp/features/memory/recompute.py`

`recompute_modules()` 同样按 plan 匹配 module，将 forward 替换为 `torch.utils.checkpoint.checkpoint` wrapper。非 reentrant 路径直接执行：

```python
checkpoint(function, *args, use_reentrant=False, **ckpt_kwargs, **kwargs)
```

因为 ChunkMBS 位于外层，所以每个 `forward_func(micro_args, micro_kwargs)` 都建立一个独立 checkpoint。forward 不保留层内所有中间激活；backward 对对应 chunk 重算原始 layer forward。这正是设备侧层内动态激活能按 `chunk_mbs` 控制的原因。

## 11. async activation offload 的配合实现

文件：`mindspeed_mm/fsdp/features/memory/async_offload.py`

legacy offload wrapper 从位置参数 0 获取 `hidden_states`，在 forward 外建立 `async_save_on_cpu` 的 `saved_tensors_hooks` 上下文。其自定义检查只选择与入口 hidden states 具有相同 `data_ptr()` 的保存 Tensor：

```python
custom_check_fn=lambda x: x.data_ptr() == hidden_states.data_ptr()
```

pack hook 的主要过程：

1. 为 Tensor 分配 pinned CPU buffer；
2. 在 swap stream 上非阻塞 D2H copy；
3. 等待适当时机后将原设备 storage resize 为 0；
4. 保存 `SwapTensor` 描述符供 backward 使用。

unpack hook 在 backward：

1. 恢复设备 storage 大小；
2. 在 swap stream 上 H2D copy；
3. 当前计算 stream 等待 H2D event；
4. 按 layer/tensor key 清理并预取下一批需要的激活。

ChunkMBS 对每块分别调用该 wrapper，所以每块入口激活分别进入 offload 生命周期。末层被特意跳过 D2H，以避免紧接 backward 前的不必要传输。

新的 `activation_offload_plan.impl: stash` 走 `act_stash.py` 与共享 SwapCore，但安装顺序仍然是 recompute → activation offload → ChunkMBS，外层切分原则不变。

## 12. FSDP2 参数生命周期

文件：`mindspeed_mm/fsdp/distributed/fully_shard_parallel.py`

配置匹配的 block 会调用 PyTorch/MindSpeed `fully_shard(module, ...)`。默认配置还把 `reshard_after_forward` 传给 FSDP。可以把单次目标 module 调用理解为：

```text
FSDP pre-forward: unshard block params
  ChunkMBS wrapper:
    chunk 0 -> offload -> checkpoint -> original forward
    chunk 1 -> offload -> checkpoint -> original forward
    ...
    cat outputs
FSDP post-forward: optional reshard
```

如果不用 ChunkMBS，而由训练调度器用 `K` 次梯度累积处理同样数据，这将变成 `K` 次独立 module 调用，每次都可能触发自己的 FSDP 参数生命周期。ChunkMBS 的通信收益来自把 `K` 次层调用折叠到一次层调用内部。

反向重计算阶段，仓库还为 checkpoint-wrapped block 的嵌套 FSDP unit 配置 `hook_module`，用于确保重计算时嵌套单元正确 unshard。它是 FSDP2 与 recompute 的兼容机制，不属于 ChunkMBS 的切分算法本身。

## 13. Qwen3.5 35B 实例

文件：`examples/qwen3_5/qwen3_5_35B_config.yaml`

该配置：

- recompute 覆盖 visual blocks 和 language model layers；
- activation offload 覆盖同一集合；
- ChunkMBS 只覆盖 language model layers；
- `training.micro_batch_size=4`；
- `chunk_mbs=2`、`batch_dim=0`；
- 位置参数 0 与四个 batch 相关 kwargs 被切分。

对每个 language layer，执行形态是（`B=4`、`C=2`）：

```text
一次 FSDP layer 调用
  ├─ forward chunk 0
  └─ forward chunk 1
cat -> 完整 batch output
```

仓库 ST 配置 `tests/st/run_configs/finetune_qwen3_5_35B/qwen3_5_35B_config.yaml` 使用同样的特性参数，训练 6 iter；但代码库当前没有针对 `_slice_batch_recursive`、输出结构和梯度等价性的独立 UT。

## 14. 建议补充的工程校验

如果要继续增强实现，优先级较高的校验与测试包括：

1. 配置期校验 `chunk_mbs` 为正整数，至少存在一个切分输入；
2. forward 前校验索引/关键字存在、Tensor 维度合法、所有 batch size 一致；
3. 匹配结果去重，并拒绝祖先/后代 module 的危险重叠包装；
4. 输出使用递归拼接协议，或明确支持 Hugging Face `ModelOutput`、dict、`None` 和非 batch 辅助量；
5. UT 覆盖整除/非整除、`batch_dim=1`、嵌套输入、Tensor/tuple/list 输出；
6. 比较未切分与切分路径的 forward、输入梯度和参数梯度；
7. 集成测试统计 FSDP unshard 次数、峰值设备显存、Host 内存和 step time，而不只比较最终 loss。

这些缺口不否定当前 Qwen3.5 已验证路径，但决定了将 ChunkMBS 移植到新模型时需要先审查 forward 签名和输出结构，不能只复制 YAML 开关。
