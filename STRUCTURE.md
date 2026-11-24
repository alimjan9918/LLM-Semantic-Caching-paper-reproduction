# 当前代码结构

```text
code/
  algorithms/
    base.py                 offline/online 公共 policy 接口
    registry.py             六种算法注册与默认五算法列表
    lazy_density_greedy.py   Lazy density greedy + best singleton oracle
    reverse_greedy.py        Memory-budget reverse greedy
    clcb_sc.py               CLCB-SC
    cucb_sc.py               CUCB-SC
    epsilon_greedy.py        Epsilon-Greedy
    lfu.py                   LFU
    ombsc.py                 OMBSC stages 与 cache switch
    vsocb.py                 VSOCB-Semantic 推荐等待机制
    knapsack_oracle.py       VSOCB 的精确可加背包
  experiments/e1.py          固定预算 online 实验与绘图
  e1.py                      实验命令入口
  online.py                  服务、反馈、大小记录、switching accounting
  objective.py               Semantic loss、gain、nearest distance
  storage.py                 Query-plus-answer storage
  estimation.py              成本噪声与 OMBSC confidence
  evaluation.py              Exact semantic optimum，仅用于评估
  data.py                    参考数据与 arrival distribution
  reference_data/            20-query 距离、token 数和 query 文本
  tests/                     当前接口测试及数值回归样例
  run_checks.py              测试入口
  verify_results.py          保存结果的预算、时序、成本和 regret 检查
  results/                   最新五算法实验
```

## 模式接口

所有比较 policy 均通过同一入口创建：

```python
from algorithms.registry import create_policy
from online import run_online_agent

policy = create_policy("CLCB-SC", mode="online", seed=0)
agent = policy.make_agent(problem, storage, horizon=200, delta=1 / 200)
result = run_online_agent(problem, storage, 200, 1 / 200, seed=0,
                          agent=agent, paired_noise=True)
```

离线模式从 supplied log 一次选择缓存：

```python
policy = create_policy("CLCB-SC", mode="offline", seed=0)
cache = policy.fit(stream, observed_costs, observed_mask,
                   distance_matrix, recorded_sizes, memory_budget, delta=0.1)
```

`observed_mask` 区分未观测值和真正的零成本观测；未知大小可用 NaN 表示。只有已获得答案的 query 才能进入候选集。Offline 模式不能创建在线 agent，online 模式不能调用 fit。

| Policy | 估计与缓存选择 | Online 更新 |
|---|---|---|
| OMBSC | Cost LCB + lazy density greedy/best singleton | Stage 触发 |
| CLCB-SC | Cost LCB + budget reverse greedy | 每轮服务后 |
| Epsilon-Greedy | Mean cost + epsilon-random reverse-greedy 删除 | 每轮服务后 |
| LFU | 按经验频率可行装箱，mean cost 决定服务动作 | 每轮服务后 |
| CUCB-SC | Cost UCB + budget reverse greedy | 每轮服务后 |
| VSOCB-Semantic | 论文成本/概率 LCB + additive knapsack + recommend-and-wait | 样本数或时间乘法增长触发 oracle |

Reverse greedy 按“删除后的 loss 增量 / 释放 memory”最小者删除，直到满足预算。Epsilon-Greedy 在需要删除时以 epsilon 概率随机删除；它并非以 epsilon 概率直接调用 LLM。LFU 会跳过放不下的项，并按 query index 处理平局。

`experiments/` 为后续 e2、e3 留出位置，目前实现 e1。所有运行命令及输出含义见 [README.md](README.md)。
