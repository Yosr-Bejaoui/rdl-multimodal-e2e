# rdl-e2e-multimodal

**An open-source reference implementation of end-to-end multimodal Relational Deep Learning** —
jointly training the text encoder with the heterogeneous GNN, instead of freezing it as a
preprocessing step.

## The gap this addresses

["Relational Deep Learning: Graph Representation Learning on Relational Databases"](https://arxiv.org/abs/2312.04615)
(Fey et al., 2023) freezes pretrained text/image encoders and trains only the GNN on top of their
embeddings, for cost and stability reasons at scale. The paper explicitly lists **end-to-end
training of the encoders** as future work. As of this writing, there is no open-source baseline
implementation of that extension. This repo is one.

## What's implemented

- **Generic relational → heterogeneous graph construction** (`rdl_e2e/data.py`): row → node,
  foreign key → edge (+ reverse), for real RelBench databases
  [RelBench](https://relbench.stanford.edu/).
- **A frozen/trainable switch on the text encoder** (`rdl_e2e/encoders.py`): one flag
  (`trainable_text`) controls whether gradients flow into the transformer or stop at the
  embedding, with everything else in the model held identical.
- **A schema-agnostic heterogeneous GNN** (`rdl_e2e/model.py`): `HeteroConv` + `SAGEConv` per
  edge type, built dynamically from `graph.metadata()`, so the same model class runs on any
  table/foreign-key schema.
- **A training + evaluation harness** (`rdl_e2e/train.py`, `rdl_e2e/evaluate.py`) with early
  stopping, LR scheduling, and metrics selected per task type (classification vs. regression),
  matching what RelBench itself reports.
- **A config-driven CLI** (`rdl_e2e/cli.py`) and a benchmark runner (`scripts/`) that runs both
  modes across every config and produces a comparison table.

## What's *not* yet done (see [Status](#status))

Real RelBench data and real pretrained weights both require network access this repo was
initially built without (see [Status](#status)). The code paths for both are implemented and
tested against RelBench data; running them for real is the immediate next step, on a machine
with internet access.

## Quickstart

```bash
git clone https://github.com/Yosr-Bejaoui/rdl-e2e-multimodal.git
cd rdl-e2e-multimodal
pip install -e ".[dev]"
pytest -v                    # smoke tests, synthetic data, no network needed

# Run a real RelBench benchmark (frozen vs. e2e):
python -m rdl_e2e.cli --config configs/relbench_rel_avito.yaml --mode both
```

### Running on real RelBench data

```bash
pip install -e ".[relbench]"   # adds relbench + pytorch_frame
python -m rdl_e2e.cli --config configs/relbench_rel_avito.yaml --mode both
```

This downloads and caches the dataset on first run (`~/.cache/relbench`) and downloads pretrained
`distilbert-base-uncased` weights on first run (`~/.cache/huggingface`). Both require outbound
network access. Use `configs/relbench_rel_f1.yaml` if you want a smaller real benchmark.

### Running the full benchmark suite

```bash
bash scripts/run_all_benchmarks.sh
python scripts/make_report.py   # -> results/benchmark_table.md
```

## Architecture

```
Relational DB (tables + foreign keys)
  │  RelationalDatabase.from_relbench()
        ▼
Heterogeneous entity graph (row→node, FK→edge)     rdl_e2e/data.py
        │
        ▼
Per-node-type encoders                              rdl_e2e/encoders.py
  ├─ TabularEncoder  (numeric + categorical)
  └─ TextEncoder     (frozen OR trainable — the switch)
        │
        ▼
N-layer HeteroConv GNN (SAGEConv per edge type)      rdl_e2e/model.py
        │
        ▼
Prediction head → logits
        │
        ▼
Loss ─┬─ frozen mode:    gradient stops before TextEncoder
      └─ e2e mode:       gradient flows through TextEncoder too
```

## Design decisions worth knowing about

- **No dependency on `torch_frame`'s stype-encoded feature pipeline.** RelBench's own examples
  pre-embed text columns as a preprocessing step via `torch_frame` — that's the frozen-encoder
  pattern this project exists to move past. `rdl_e2e/data.py` keeps raw text as text through graph
  construction; `rdl_e2e/encoders.py` decides frozen vs. trainable per run.
- **Column types are inferred generically** (`infer_column_types` in `data.py`) from dtype and
  string length/cardinality, rather than requiring a schema file, so the same code runs on any
  RelBench dataset without per-dataset wiring.
- **One config file = one experiment.** `configs/*.yaml` fully specify dataset, model, and
  training hyperparameters, so `frozen` and `e2e` runs are guaranteed identical except for the one
  flag that matters.

## Status

Built in an environment without outbound access to `huggingface.co` or `relbench.stanford.edu`.
Concretely:

- `RelationalDatabase.from_relbench(...)` calls the real `relbench` API and is untested against
  live data (blocked at build time: `HTTPError 403` from `relbench.stanford.edu`).
- `Tokenizer` and `TextEncoder` try to load real pretrained weights first and only fall back to a
  randomly-initialized same-architecture model with a loud `warnings.warn(...)` if that fails —
  this fallback is what allows the synthetic tests/demo to run anywhere, but it means results
  produced this way demonstrate the training *mechanics*, not the benefit of fine-tuning genuine
  pretrained knowledge.
- Everything else (graph construction, GNN, training loop, evaluation, CLI) is fully implemented
  and passes `pytest` against the synthetic dataset.

**To take this from "reference implementation" to "baseline with real numbers":** run
`scripts/run_all_benchmarks.sh` with `configs/relbench_*.yaml` on a machine with internet access,
then fill in `results/benchmark_table.md`, then update this section.

## Citation

If you use this code, please cite the original paper this extends:

```bibtex
@article{fey2023relational,
  title={Relational Deep Learning: Graph Representation Learning on Relational Databases},
  author={Fey, Matthias and Hu, Weihua and Huang, Kexin and Lenssen, Jan Eric and Ranjan, Rishabh
          and Robinson, Joshua and Ying, Rex and You, Jiaxuan and Leskovec, Jure},
  journal={arXiv preprint arXiv:2312.04615},
  year={2023}
}
```

## License

MIT — see [LICENSE](LICENSE).
