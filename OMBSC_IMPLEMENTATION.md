# OMBSC online implementation

This project replaces the earlier per-round online LCB/UCB experiments with
the online algorithm from *Semantic Caching under Memory Budgets for LLM
Serving*.

## Algorithm mapping

| Paper component | Implementation |
|---|---|
| Algorithm 1: OMBSC main loop | `OMBSC.step` |
| Algorithm 2: `SwitchCache` | `OMBSC._switch_cache` |
| Equation (4): optimistic cost | `ombsc_confidence_costs` |
| Algorithm 3: lazy greedy oracle | `memory_budget_greedy_oracle` |
| Query-plus-answer storage | `OnlineStorageModel` |
| One complete online run | `run_ombsc` |

## Information boundary

For every query `q`:

```text
L(q) = known query size
A(q) = hidden response size
S(q) = L(q) + A(q)
```

The agent only writes `S(q)` into its known-size array after it invokes the LLM
for `q` for the first time. Only revealed queries enter the candidate set `V`.

The OMBSC confidence radius uses the paper's time-union term `T^3`:

```text
radius = SIGMA * sqrt(
    2 * log(4 * N * T^3 / delta) / max(N_c(q), 1)
)
```

This is intentionally separate from the original author reproduction helper
`caching.confidence_radius`, which uses `T`.

The oracle may evaluate semantic coverage over all of `Q`, but it can select
only entries from `V`.

## Switching cost

If an update changes the cache from `M_t` to `M_{t+1}`, the new entries incur

```text
K_{t+1} = sum_{q in M_{t+1} \ M_t} c(q).
```

The cost is charged in the next online round. `OMBSCResult.cumulative_objective`
contains `sum(loss) + sum(K_t)`.

## Run

```powershell
.\.venv\Scripts\python.exe run_memory_budgeted.py --quick
.\.venv\Scripts\python.exe run_memory_budgeted.py --runs 10
```

Results are written under `results/ombsc/` when using:

```powershell
.\.venv\Scripts\python.exe run_memory_budgeted.py --runs 10 --out results\ombsc
```

## Scope

The code faithfully simulates the online information flow and memory budget.
It still uses a synthetic response-size proxy rather than allocating and
storing complete LLM responses in real CPU or GPU memory.
