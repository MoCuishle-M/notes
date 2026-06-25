# FSDP2 代码架构梳理

## 必须遵守的规则

1. 必须看源码，fsdp2函数入口是

```python
from torch.distributed.fsdp import fully_shard
```

1. 请优先参考我提供的官方文档内容来校准你的知识库。

pytorch官方文档：[https://docs.pytorch.org/docs/2.12/distributed.fsdp.fully_shard.html](https://docs.pytorch.org/docs/2.12/distributed.fsdp.fully_shard.html)。
pytorch官方fsdp2基础教程：[https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html?spm=5176.28103460.0.0.7b0b2988mpFKNO](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html?spm=5176.28103460.0.0.7b0b2988mpFKNO)。

## Role

你是一位精通 PyTorch 底层分布式系统与 Python 交互的高级架构师。

## Context

我正在深入阅读 PyTorch FSDP2 (Fully Sharded Data Parallelism v2) 的源码实现，目标是彻底搞懂它的实际代码结构和底层实现机制。
**核心界定**：我关注的是基于 **DTensor** 和 **Composable API** 的新版 FSDP2（入口为 `fully_shard`），**绝对不是**基于 FlatParameter 和 Hooks 的旧版 FSDP1（`FullyShardedDataParallel`）。

## Constraints

1. **禁止大段源码**：只输出结构、概括、调用链和核心逻辑，不要粘贴大段 Python/C++ 源码。
2. **图表规范**：所有的架构图、依赖图、时序图，**必须使用 Mermaid 语法**输出，以便我直接渲染。
3. **精准定位**：FSDP2 的核心代码主要位于 `torch\distributed\fsdp\_fully_shard\` 目录下，请基于此真实路径进行梳理，切勿与 FSDP1 混淆。

## Tasks

请帮我做一个高层次的代码架构梳理，具体分为以下四个模块。输出对应的文档放在fsdp2\docs目录下，文件要命名清楚。

### 模块一：宏观目录与文件结构

1. 列出 FSDP2 核心代码在 PyTorch 仓库中的主要目录和关键 Python 文件路径。
2. 用**一句话**精准概括每个关键文件的作用。
3. 使用 Mermaid 绘制一个**模块依赖树状图**，说明这些文件/模块之间的依赖与组织关系。

### 模块二：核心入口与架构演进

1. FSDP2 的核心入口函数 `fully_shard` 的完整调用链是怎样的？（从用户调用到最终完成参数 Sharding 和 Hook 注册）。
2. 对比 FSDP1，FSDP2 在底层架构上做了哪些**根本性改变**？（请重点从 FlatParameter vs DTensor、内存管理机制 recordStream、Hook 机制 vs Subclass/Tensor 扩展点 等维度进行对比，使用表格形式呈现）。

### 模块三：核心类与职责拆解

列出 FSDP2 实现中最核心的 4-6 个 Python 类（例如 `FSDPState`, `FSDPParam`, `FSDPParamGroup` 等），并使用表格说明：

- 类名及所在文件
- 核心职责
- 管理的关键数据结构（如 DTensor, ShardedTensor 等）

### 模块四：运行时生命周期与调用时序

请按照 FSDP2 的实际运行生命周期，梳理各模块的调用关系：

1. **Initialization (Sharding)**：参数如何被转换为 DTensor 并切分。
2. **Forward Pass**：Pre-forward (All-gather/Unshard) -> Compute -> Post-forward (Reshard)。
3. **Backward Pass**：Pre-backward (All-gather) -> Compute -> Post-backward (Reduce-scatter)。
4. **Prefetching 机制**：隐式/显式 Prefetch 是如何在 CPU 线程和 CUDA Stream 之间调度的？

最后，请使用 Mermaid 绘制一张 **Forward + Backward 过程中的核心模块调用时序图 (Sequence Diagram)**，需体现出 CPU 线程、CUDA 计算流、通信流之间的交互。
