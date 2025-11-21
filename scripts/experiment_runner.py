#!/usr/bin/env python3
"""
scripts/experiment_runner.py

Orchestrate a backtest experiment over a date range using existing snapshot builder,
publish_feed, postprocess_v3, and update_performance_v3 in dry-run mode.

Usage (example):
  python scripts/experiment_runner.py --start 2023-01-01 --end 2023-02-28 --step 5 --experiment exp_v1

This script:
 - builds per-date snapshots using scripts/build_snapshot_feed_from_backfill.py
 - runs publish_feed.py (BACKTEST_MODE=1) to compute TS/NS/CDS
 - runs postprocess_v3.py to produce recommended trades
 - runs update_performance_v3.py in DRY-RUN (APPLY_TRADES=0) to compute history rows
 - After the loop it calls analyze_backtest_experiment.py to produce analytics

This runner is intended to be called from CI (GitHub Actions) or locally.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, json, yaml, shutil
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run_cmd(env, *cmd):
    """Run a command with env updates, stream output."""
    e = os.environ.copy()
    e.update(env or {})
    print(f"[RUN] {' '.join(cmd)} (env overrides: {list(env.keys()) if env else []})")
    r = subprocess.run(cmd, env=e)
    if r.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd)} (rc={r.returncode})")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    ap.add_argument("--step", default="5", help="Step days between snapshots")
    ap.add_argument("--experiment", default="exp_v1", help="Experiment id")
    ap.add_argument("--portfolios_tag", default="backtest", help="Portfolio tag to include")
    return ap.parse_args()


def main():
    args = parse_args()
    start = datetime.fromisoformat(args.start).date()
    end = datetime.fromisoformat(args.end).date()
    step_default = int(args.step)
    exp = args.experiment
    tag = args.portfolios_tag

    out_dir = ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # support variants file: JSON list of variant dicts
    variants_file = ROOT / 'experiments' / 'variants.json'
    if variants_file.exists():
        print(f"[INFO] Found variants file at {variants_file}; running each variant")
        variants = json.loads(variants_file.read_text(encoding='utf-8'))
    else:
        variants = [
            {"name": "default", "backtest_half_life_hours": os.environ.get('BACKTEST_HALF_LIFE_HOURS','336'), "backtest_ns_max_age_days": os.environ.get('BACKTEST_NS_MAX_AGE_DAYS','90'), "step_days": step_default}
        ]

    # Read base portfolios config to allow variant overrides
    base_config_path = ROOT / 'config' / 'portfolios.yml'
    base_config = {}
    if base_config_path.exists():
        base_config = yaml.safe_load(base_config_path.read_text(encoding='utf-8')) or {}

    # Determine list of portfolios with the requested tag
    portfolios_cfg = base_config.get('portfolios', {}) if isinstance(base_config, dict) else {}
    selected_portfolios = [pid for pid, p in portfolios_cfg.items() if 'backtest' in [(t or '').lower() for t in (p.get('tags') or [])]]

    # Ensure variant portfolio positions exist by copying base positions if missing
    base_positions = ROOT / 'data' / 'portfolios' / 'port_backfill_5each' / 'positions.csv'
    for v in variants:
        # create per-variant portfolio directories for each selected portfolio
        for pid in selected_portfolios:
            variant_port_name = pid + '_' + v.get('name')
            target_dir = ROOT / 'data' / 'portfolios' / variant_port_name
            target_dir.mkdir(parents=True, exist_ok=True)
            target_positions = target_dir / 'positions.csv'
            if not target_positions.exists() and base_positions.exists():
                import shutil
                shutil.copy2(base_positions, target_positions)
                print(f"[INFO] copied base positions to {target_positions}")

    # Run each variant sequentially
    for v in variants:
        name = v.get('name')
        run_id = f"{exp}_{name}"
        hl = str(v.get('backtest_half_life_hours', os.environ.get('BACKTEST_HALF_LIFE_HOURS','336')))
        max_age = str(v.get('backtest_ns_max_age_days', os.environ.get('BACKTEST_NS_MAX_AGE_DAYS','90')))
        step = int(v.get('step_days', step_default))

        print(f"[INFO] Running variant {name} -> experiment id {run_id} (hl={hl}h, max_age={max_age}d, step={step}d)")

        current = start
        i = 0
        while current <= end:
            as_of = current.isoformat()
            print(f"=== {name} STEP {i} as_of={as_of} ===")

            # Build snapshot with variant env
            env = {"BACKTEST_NS_MAX_AGE_DAYS": max_age, "BACKTEST_HALF_LIFE_HOURS": hl}
            run_cmd(env, PY, str(ROOT / "scripts" / "build_snapshot_feed_from_backfill.py"),
                    "--as-of", as_of,
                    "--input", str(ROOT / "data" / "prices_backfill.json"),
                    "--output", str(ROOT / "data" / "prices_final_snapshot.json"))

            # publish
            env = {"BACKTEST_MODE": "1", "SNAPSHOT_AS_OF": as_of,
                   "INPUT_PATH": str(ROOT / "data" / "prices_final_snapshot.json"),
                   "OUTPUT_PATH": str(ROOT / "data" / "prices_final.json"),
                   "BACKTEST_HALF_LIFE_HOURS": hl}
            run_cmd(env, PY, str(ROOT / "publish_feed.py"))

            # Write a temporary portfolios.yml that points portfolios to per-variant positions
            tmp_cfg = ROOT / '.tmp' / f'portfolios_{run_id}.yml'
            tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
            base_cfg_path = ROOT / 'config' / 'portfolios.yml'
            if base_cfg_path.exists():
                base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding='utf-8')) or {}
                cfg_copy = dict(base_cfg)
                # modify portfolios paths for selected portfolios
                for pid in selected_portfolios:
                    node = cfg_copy.get('portfolios', {}).get(pid)
                    if not node:
                        continue
                    variant_port_name = pid + '_' + name
                    node['path'] = str(ROOT / 'data' / 'portfolios' / variant_port_name)
                    # ensure tags contain variant label
                    tags = node.get('tags', []) or []
                    if f'variant:{name}' not in tags:
                        tags.append(f'variant:{name}')
                    node['tags'] = tags
                    cfg_copy['portfolios'][pid] = node
                tmp_cfg.write_text(yaml.safe_dump(cfg_copy), encoding='utf-8')
                print(f"[INFO] wrote temp portfolios config {tmp_cfg}")
            else:
                tmp_cfg = None

            # postprocess (use temp portfolios config if present)
            post_env = {"INPUT_FINAL": str(ROOT / "data" / "prices_final.json")}
            if tmp_cfg:
                post_env['PORTFOLIOS_YML'] = str(tmp_cfg)
            run_cmd(post_env, PY, str(ROOT / "scripts" / "postprocess_v3.py"))

            # update_performance in dry-run or apply based on env
            apply_flag = os.environ.get('APPLY_TRADES','0')
            perf_env = {"BACKTEST_MODE": "1",
                   "SNAPSHOT_AS_OF": as_of,
                   "BACKTEST_PRICES_FINAL": str(ROOT / "data" / "prices_final.json"),
                   "BACKTEST_TRADES_PATH": str(ROOT / "artifacts" / "recommended_trades_v3.json"),
                   "BACKTEST_TRADES_LEDGER": str(ROOT / "data" / "trades_ledger_backtest.csv"),
                   "APPLY_TRADES": apply_flag}
            # ensure update_performance reads the same portfolios config so positions paths are consistent
            if tmp_cfg:
                perf_env['PORTFOLIOS_YML'] = str(tmp_cfg)
            run_cmd(perf_env, PY, str(ROOT / "scripts" / "update_performance_v3.py"))

            # advance
            current = current + timedelta(days=step)
            i += 1

        # After finishing all snapshots for this variant, copy per-variant portfolio outputs into the experiment artifact folder once
        variant_art_root = out_dir / run_id / 'portfolios'
        variant_art_root.mkdir(parents=True, exist_ok=True)
        for pid in selected_portfolios:
            variant_port_name = pid + '_' + name
            src = ROOT / 'data' / 'portfolios' / variant_port_name
            dst = variant_art_root / variant_port_name
            try:
                if src.exists():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    print(f"[INFO] copied {src} -> {dst}")
            except Exception as e:
                print(f"[WARN] failed to copy {src} -> {dst}: {e}")

        # After variant loop, call analyzer for this variant across all backtest portfolios
        cfg_for_analyzer = str(tmp_cfg) if tmp_cfg else str(ROOT / "config" / "portfolios.yml")
        analyzer_env = {
            "EXPERIMENT_ID": run_id,
            "CONFIG_PORTFOLIOS": cfg_for_analyzer,
            "BENCHMARK_SYMBOL": os.environ.get("BENCHMARK_SYMBOL", "SPY"),
        }
        run_cmd(analyzer_env, PY, str(ROOT / "scripts" / "analyze_backtest_experiment.py"), str(ROOT / "data" / "prices_backfill.json"))

    print(f"[OK] experiment runner completed. artifacts in {out_dir}")

if __name__ == '__main__':
    main()
