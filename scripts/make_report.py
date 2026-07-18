import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name == "benchmark_table.md":
            continue
        runs = json.loads(path.read_text())
        by_mode = {r["mode"]: r for r in runs}
        if "frozen" not in by_mode or "e2e" not in by_mode:
            continue
        frozen, e2e = by_mode["frozen"], by_mode["e2e"]
        primary_metric = next(iter(frozen["metrics"]))
        rows.append({
            "config": path.stem,
            "dataset": frozen["dataset"],
            "metric": primary_metric,
            "frozen": round(frozen["metrics"][primary_metric], 4),
            "e2e": round(e2e["metrics"][primary_metric], 4),
            "delta": round(e2e["metrics"][primary_metric] - frozen["metrics"][primary_metric], 4),
            "frozen_time_s": round(frozen["train_time_sec"], 2),
            "e2e_time_s": round(e2e["train_time_sec"], 2),
            "frozen_params": frozen["trainable_params"],
            "e2e_params": e2e["trainable_params"],
        })

    lines = [
        "| Config | Dataset | Metric | Frozen | E2E | Δ | Frozen time (s) | E2E time (s) | Frozen params | E2E params |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['dataset']} | {r['metric']} | {r['frozen']} | {r['e2e']} | "
            f"{r['delta']:+.4f} | {r['frozen_time_s']} | {r['e2e_time_s']} | {r['frozen_params']} | {r['e2e_params']} |"
        )

    out_path = RESULTS_DIR / "benchmark_table.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
