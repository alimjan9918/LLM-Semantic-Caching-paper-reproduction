# Memory-Budgeted Semantic Caching


Online semantic caching for one fixed LLM with unknown query frequencies and generation costs, heterogeneous query–answer sizes, and cache switching costs. Experiments replay recorded data; no generative LLM or paid API is required.

### Method and structure

OMBSC records arrival counts and fresh-call feedback, then refreshes estimates and selects a cache at stage boundaries. Its oracle uses **lazy density greedy + best feasible singleton** under `sum S(q) <= B`. Only candidates with observed answer sizes are eligible. Lazy selection recomputes the heap-top marginal loss reduction per byte and accepts it when it exceeds the remaining stored upper bounds; otherwise it reinserts it. Semantic coverage is evaluated over the full query set.

```text
algorithms/          Independent policies and cache-selection oracles
experiments/         e1/e2 execution, evaluation, and plotting
reference_data/      Bundled 20-query example
real-world-dataset/  Local RouterBench data, embedding model, and vector cache
online.py            Service decisions, feedback, and switching accounting
objective.py         Semantic loss and loss reduction
storage.py           Query + answer storage and revealed sizes
estimation.py        Cost noise and confidence estimates
evaluation.py        Exact semantic benchmark
data.py              Example data and arrival distributions
real_world.py        Recorded-data loading and CPU embeddings
preprocessing.py     Extract one model's records into CSV
download_real_world.py  Download data and embedding files
e1.py / e2.py        Online experiment entry points
run_checks.py        Tests
results/             Generated outputs
```

| File in `algorithms/` | Method / reference |
|---|---|
| `ombsc.py` | Our stage-triggered learning framework; builds on the low-switching design in [1] |
| `lazy_density_greedy.py` | Memory-constrained density greedy with lazy evaluation and singleton safeguard; lazy selection follows the idea in [3], Algorithm 2 |
| `reverse_greedy.py` | Budget adaptation of the semantic-cache selector in [1] |
| `clcb_sc.py`, `cucb_sc.py` | Cost-LCB / cost-UCB semantic caching, adapted from [1] |
| `epsilon_greedy.py` | Mean-cost reverse greedy with epsilon-random deletions |
| `lfu.py` | Frequency-based feasible packing |
| `vsocb.py` | VSOCB from [2], adapted to semantic mismatch costs |
| `knapsack_oracle.py` | Exact additive knapsack for VSOCB |
| `min_knapsack_oracle.py` | Complement oracle from [2], Appendix A; minimum-knapsack solver based on [4] |
| `base.py`, `registry.py` | Shared offline/online policy interfaces and registration |

Baselines share our memory-budget, mismatch-cost, and switching-cost setting. They are adaptations, not exact reproductions with unchanged theoretical guarantees. Lazy greedy is an established technique; [3] optimizes a different objective.

### Install and prepare data

Run commands from the repository root (the directory containing `e1.py`):

```bash
python -m pip install -r requirements.txt
python run_checks.py
python e1.py --quick --out results/e1_demo
```

For e2, download RouterBench [5] and the pinned CPU embedding model, then extract one model:

```bash
python download_real_world.py
python preprocessing.py --list-models
python preprocessing.py --model gpt-3.5-turbo-1106
```

Alternatively, download these files manually:

- [routerbench_0shot.pkl](https://huggingface.co/datasets/withmartian/routerbench/resolve/784021482c3f320c6619ed4b3bb3b41a21424fcb/routerbench_0shot.pkl?download=true) → `real-world-dataset/routerbench_0shot.pkl`
- [tokenizer.json](https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/751bff37182d3f1213fa05d7196b954e230abad9/tokenizer.json?download=true) → `real-world-dataset/embedding-model/tokenizer.json`
- [model_quantized.onnx](https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/751bff37182d3f1213fa05d7196b954e230abad9/onnx/model_quantized.onnx?download=true) → `real-world-dataset/embedding-model/onnx/model_quantized.onnx`

Preprocessing creates `real-world-dataset/routerbench_single_model.csv`. Embeddings are computed and cached automatically on the first e2 run. Keep all usable records during preprocessing if you plan to vary `--queries` later.

### Run and configure

```bash
# Uniform arrivals, default five algorithms
python e2.py --queries 40 --horizon 5000 --runs 5 --arrival uniform --budget-fraction 0.25 --out results/e2_uniform

# Zipf arrivals, fixed byte budget, selected algorithms
python e2.py --queries 100 --horizon 5000 --runs 5 --arrival zipf --budget 10000 --algorithms OMBSC VSOCB-Semantic --vsocb-oracle auto --out results/e2_zipf

# Tune OMBSC confidence without changing environmental noise
python e2.py --queries 40 --horizon 5000 --runs 5 --seed 5 --data-seed 0 --noise-sigma 0.05 --ombsc-confidence-scale 0.25 --out results/e2_tuned

python e1.py --help
python e2.py --help
```

| Parameter | Default | Meaning |
|---|---|---|
| `--queries` | 20 | Number of distinct queries; e1 supports 2–20 |
| `--horizon`, `--runs` | e1: 200, 10; e2: 1000, 5 | Rounds and repeated trials |
| `--quick` | off | Overrides horizon/runs with 100/2 |
| `--arrival` | uniform | `uniform`: equal probabilities; `zipf`: probabilities proportional to rank^(-1.1); exponent currently fixed |
| `--seed` | 0 | First trial seed; runs use consecutive seeds |
| `--budget-fraction` | 0.25 | Fraction of total query–answer storage |
| `--budget` | unset | Absolute budget; mutually exclusive with budget-fraction; e1 uses normalized units, e2 uses UTF-8 bytes |
| `--algorithms` | OMBSC CLCB-SC Epsilon-Greedy LFU VSOCB-Semantic | Space-separated names; CUCB-SC is also available |
| `--ombsc-confidence-scale` | 1 | OMBSC confidence-radius multiplier; must be positive; values below 1 are empirical tuning without the original confidence guarantee |
| `--epsilon` | 0.15 | Epsilon-Greedy random-deletion probability, not LLM-call probability |
| `--vsocb-growth` | 1 | VSOCB trigger growth factor is 1 + this value |
| `--vsocb-oracle` | auto | `auto`: exact with <=40 feasible candidates, approximate otherwise; `exact` and `approx` force the corresponding solver |
| `--alpha` | (1-1/e)/2 | Regret evaluation coefficient; does not change algorithm actions |
| `--out` | results/e1 or results/e2 | Output directory; existing same-name files are overwritten |
| `--data-seed` (e2) | 0 | Query-subset sampling seed, distinct from trial seed |
| `--noise-sigma` (e2) | 0.05 | Environment noise bound and related confidence setting |
| `--gamma` (e2) | 1 | Semantic mismatch-cost scale |
| `--benchmark` (e2) | auto | `auto`: exact semantic optimum for N<=22, offline greedy reference otherwise; can force `exact` or `greedy` |
| `--data-dir` (e2) | real-world-dataset | Data/model/cache directory |
| `--dataset` (e2) | DATA_DIR/routerbench_single_model.csv | Preprocessed CSV path |
| `--model`, `--eval-name` (e2) | unset | Check the CSV model; filter by exact eval_name |

Both entry points run online experiments; offline policies are available through the Python API. e1 fixes sigma=0.05 and gamma=1. In e2, the query subset is fixed across trials. Zipf ranks follow the selected query order, not measured popularity. Arrival probabilities are simulated and hidden from the algorithms.

Tune confidence scale on validation seeds, then evaluate on separate seeds. Keep noise, gamma, budget, and workload fixed when comparing algorithm parameters. Stage thresholds currently have no CLI multiplier.

### Outputs

Each run saves JSON configuration/traces, `summary.csv` (one row per algorithm and seed), `curves.csv`, and a PNG with means ± one standard deviation. e2 also saves `workload.json`.

The four e2 panels show: alpha-regret (or offline-reference gap), average ideal loss + switching, cumulative switching cost, and fresh service calls. e1's last panel instead shows average cache loss. For N>22 with the default e2 benchmark, the gap is to a feasible offline greedy reference, not the exact optimum.

Recorded API costs are normalized by twice the selected workload's maximum USD cost. Noise is mean-preserving: `C=c ± min(sigma,c,1-c)`. Sizes include prompt + answer bytes. Ideal expected cache loss and realized action costs are logged separately; switching is charged for newly admitted entries in the next round.


本项目研究固定单个 LLM 的在线语义缓存：query 频率和生成成本未知，query–answer 大小不同，缓存更新产生切换成本。实验回放已有数据，不需要运行生成式 LLM 或调用付费 API。

### 方法与代码结构

OMBSC 记录到达次数和 fresh-call 反馈，在阶段触发时更新估计并调用 **lazy density greedy + 最优可行 singleton**。每次只重新计算队首的单位空间边际收益，若不低于其他候选保存的上界则接受，否则重新入队。缓存满足 `sum S(q) <= B`，候选只包含已观测回答大小的 query，语义覆盖收益在全部 query 上计算。

`algorithms/` 存放独立算法和 oracle；`experiments/` 负责运行与绘图；`online.py` 处理服务、反馈和切换；`objective.py`、`storage.py`、`estimation.py` 分别处理损失、存储和估计；`evaluation.py` 计算精确参考；`real_world.py` 与 `preprocessing.py` 处理真实数据；`reference_data/` 是附带示例，`real-world-dataset/` 保存本地数据，`results/` 保存输出。

| algorithms 文件 | 方法与来源 |
|---|---|
| `ombsc.py` | 我们的阶段触发学习框架，基于 [1] 的低切换设计扩展 |
| `lazy_density_greedy.py` | 存储预算下的密度贪心和 singleton 保底；lazy 选择借鉴 [3] Algorithm 2 |
| `reverse_greedy.py` | [1] 的语义缓存选择器在存储预算下的适配 |
| `clcb_sc.py`、`cucb_sc.py` | [1] 的成本 LCB/UCB 方法适配 |
| `epsilon_greedy.py` | 经验均值 + epsilon 概率随机删除 |
| `lfu.py` | 根据经验频率进行可行装箱 |
| `vsocb.py` | [2] 的 VSOCB，适配语义失配成本 |
| `knapsack_oracle.py` | VSOCB 的精确可加背包 |
| `min_knapsack_oracle.py` | [2] Appendix A 补集方法，minimum-knapsack 子求解器采用 [4] |
| `base.py`、`registry.py` | offline/online 公共接口与算法注册 |

各 baseline 在相同预算、语义失配和切换成本下比较，适配后不直接继承原论文的理论保证。Lazy greedy 是已有技术，[3] 的原优化目标与本项目不同。

### 安装与使用

在包含 `e1.py` 的代码根目录运行：

```bash
python -m pip install -r requirements.txt
python run_checks.py
python e1.py --quick --out results/e1_demo
```

e2 先准备数据，再运行实验：

```bash
python download_real_world.py
python preprocessing.py --list-models
python preprocessing.py --model gpt-3.5-turbo-1106
python e2.py --queries 40 --horizon 5000 --runs 5 --arrival uniform --budget-fraction 0.25 --out results/e2_uniform
```

也可使用英文部分的三个链接手动下载到指定路径。预处理生成固定模型 CSV；首次 e2 自动计算并缓存 CPU embeddings。预处理时不限制条数，方便后续改变 query 数目。

```bash
# 长尾分布、固定字节预算、算法子集
python e2.py --queries 100 --horizon 5000 --runs 5 --arrival zipf --budget 10000 --algorithms OMBSC VSOCB-Semantic --vsocb-oracle auto --out results/e2_zipf

# 只调整 OMBSC 置信半径，保持环境噪声不变
python e2.py --queries 40 --horizon 5000 --runs 5 --seed 5 --data-seed 0 --noise-sigma 0.05 --ombsc-confidence-scale 0.25 --out results/e2_tuned

python e1.py --help
python e2.py --help
```

### 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--queries` | 20 | query 数目；e1 支持 2–20 |
| `--horizon`、`--runs` | e1: 200、10；e2: 1000、5 | 总轮数与重复次数 |
| `--quick` | 关闭 | 覆盖轮数和重复次数为 100、2 |
| `--arrival` | uniform | 均匀分布；或 zipf，概率正比于排名的 -1.1 次方，指数目前固定 |
| `--seed` | 0 | 起始实验种子，连续使用 runs 个种子 |
| `--budget-fraction` | 0.25 | 总 query–answer 存储量的预算比例 |
| `--budget` | 未指定 | 绝对预算，与比例参数互斥；e1 为归一化单位，e2 为 UTF-8 字节 |
| `--algorithms` | 默认五种 | 空格分隔算法名称，见英文参数表；CUCB-SC 可额外选择 |
| `--ombsc-confidence-scale` | 1 | OMBSC 置信半径的正乘数；小于 1 属于经验调优，不直接沿用原置信保证 |
| `--epsilon` | 0.15 | Epsilon-Greedy 的随机删除概率，不是调用 LLM 的概率 |
| `--vsocb-growth` | 1 | VSOCB 的触发增长因子为 1 + 此值 |
| `--vsocb-oracle` | auto | 可行候选不超过 40 时精确，否则近似；可强制 exact/approx |
| `--alpha` | (1-1/e)/2 | regret 评估系数，不改变算法行为 |
| `--out` | results/e1 或 results/e2 | 输出目录；同名文件会被覆盖 |
| `--data-seed`，仅 e2 | 0 | 抽取 query 子集的种子，区别于实验种子 |
| `--noise-sigma`，仅 e2 | 0.05 | 环境噪声上界及相关置信设置 |
| `--gamma`，仅 e2 | 1 | 语义失配成本系数 |
| `--benchmark`，仅 e2 | auto | N<=22 精确评估，否则 offline greedy 参考；可强制 exact/greedy |
| `--data-dir`，仅 e2 | real-world-dataset | 数据、模型与向量缓存目录 |
| `--dataset`，仅 e2 | DATA_DIR/routerbench_single_model.csv | 预处理 CSV 路径 |
| `--model`、`--eval-name`，仅 e2 | 未指定 | 检查 CSV 模型；按 eval_name 精确筛选 |

两个实验入口均为 online，offline 通过 Python policy API 使用。e1 固定 sigma=0.05、gamma=1。e2 各次重复使用同一 query 集合；Zipf 排名来自所选顺序，不是真实访问热度。到达流为模拟生成，算法不知道真实概率。

调参时固定数据集合、噪声、gamma 和预算，在一组种子上选择 confidence scale，再用其他种子评估。目前没有 stage threshold 倍数的命令行参数。

### 输出

输出包含配置与逐轮 JSON、每算法每种子一行的 `summary.csv`、`curves.csv` 和均值 ± 一个标准差的 PNG；e2 另有 `workload.json`。

e2 四图依次为 alpha-regret（大规模时为参考 gap）、平均理想 loss + switching、累计 switching cost、累计 fresh service 次数；e1 第四图为平均 cache loss。N>22 时默认参考为可行的 offline greedy，gap 不代表相对精确最优解的 regret。

记录的 API 费用按所选集合最大美元成本的两倍归一化，噪声为均值不变的 `C=c ± min(sigma,c,1-c)`。大小计入 prompt 和回答。理想 expected loss 与实际服务动作成本分别记录；新缓存项的切换成本在下一轮计入。

## References / 参考文献

1. Liu et al. [Semantic Caching for Low-Cost LLM Serving: From Offline Learning to Online Adaptation](https://arxiv.org/abs/2508.07675). INFOCOM 2026.
2. Yang et al. [LLM Cache Bandits with Query Heterogeneity for Cost-Efficient Inference](https://aclanthology.org/2026.emnlp-main.1849/). EMNLP 2026.
3. Perrault et al. [Budgeted Online Influence Maximization](https://proceedings.mlr.press/v119/perrault20a.html). ICML 2020, Algorithm 2 (lazy evaluation).
4. Carnes and Shmoys. [Primal-Dual Schema for Capacitated Covering Problems](https://people.orie.cornell.edu/shmoys/pdf/CarnesShmoys08.pdf). 2008, Section 2 (minimum knapsack).
5. Hu et al. [RouterBench: A Benchmark for Multi-LLM Routing System](https://arxiv.org/abs/2403.12031). 2024. [Dataset](https://huggingface.co/datasets/withmartian/routerbench).

Initial implementation / 初始实现来源: [LLM-Semantic-Caching-paper-reproduction](https://github.com/alimjan9918/LLM-Semantic-Caching-paper-reproduction), commit `275a7baf6f3647d1f0e80fdef87e40e45eff49f4`.
