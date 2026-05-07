#!/usr/bin/env python3
"""Build an operation bank: unified inventory of all proofreading operations.

Usage:
    # pass 1: collect operations
    pixi run python scripts/build_operation_bank.py build \
        --species mouse --target-count 1000 --seed 42 \
        --output datasets/mouse/operation_bank.jsonl

    # pass 1 with quality filters (human/zebrafish)
    pixi run python scripts/build_operation_bank.py build \
        --species human --target-count full \
        --min-ops-per-mm 10 --min-path-um 500 \
        --output datasets/human/operation_bank.jsonl

    # pass 2: generate inversion controls
    pixi run python scripts/build_operation_bank.py controls \
        --bank-input datasets/mouse/operation_bank.jsonl \
        --controls-output datasets/mouse/controls.jsonl

    # both passes in one command
    pixi run python scripts/build_operation_bank.py run \
        --species mouse --target-count 1000 --seed 42 \
        --output datasets/mouse/operation_bank.jsonl

    # stats
    pixi run python scripts/build_operation_bank.py stats \
        --bank-input datasets/mouse/operation_bank.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / ".." / "src"))

from connectome.utils import get_client_for_species, get_latest_proofread_roots
from connectome.proofread_roots import (
    ProofreadRootConfig,
    get_quality_filtered_roots,
)
from connectome.operation_bank import (
    OperationBankBuilder,
    generate_inversion_controls,
    bank_stats,
)


def cmd_build(args):
    species = args.species
    print(f"=== building operation bank for {species} ===")

    client = get_client_for_species(species)

    # get roots (with quality filtering for human/zebrafish)
    if species in ("human", "zebrafish") and (
        args.min_ops_per_mm is not None or args.min_path_um is not None
    ):
        config = ProofreadRootConfig(
            species=species,
            min_ops_per_mm=args.min_ops_per_mm or 20.0,
            min_path_um=args.min_path_um or 100.0,
            max_stale_days=args.max_stale_days,
            seed=args.seed,
        )
        roots = get_quality_filtered_roots(client, species, config, seed=args.seed)
    else:
        roots = get_latest_proofread_roots(client, species, seed=args.seed)

    print(f"  {len(roots)} roots available")

    target = None if args.target_count == "full" else int(args.target_count)

    output = Path(args.output)
    builder = OperationBankBuilder(
        client=client,
        species=species,
        output_path=output,
        target_count=target,
    )

    total = builder.build(roots)
    print(f"\n=== done: {total} operations in {output} ===")


def cmd_controls(args):
    bank_input = Path(args.bank_input)
    controls_output = Path(args.controls_output)

    if not bank_input.exists():
        print(f"error: bank file not found: {bank_input}")
        sys.exit(1)

    print(f"=== generating inversion controls from {bank_input} ===")
    count = generate_inversion_controls(bank_input, controls_output)
    print(f"\n=== done: {count} controls in {controls_output} ===")


def cmd_run(args):
    """Run both passes: build bank → generate inversion controls."""
    # pass 1
    cmd_build(args)

    # pass 2: controls output lives next to the bank
    bank_path = Path(args.output)
    controls_path = bank_path.with_name(
        bank_path.stem + "_controls" + bank_path.suffix
    )

    print(f"\n=== pass 2: generating inversion controls ===")
    count = generate_inversion_controls(bank_path, controls_path)
    print(f"=== done: {count} controls in {controls_path} ===")

    # quick stats
    s = bank_stats(bank_path)
    print(f"\n=== summary ===")
    print(f"  ops:      {s['total']} ({s['merges']} merges, {s['splits']} splits)")
    print(f"  controls: {count}")
    print(f"  bank:     {bank_path}")
    print(f"  controls: {controls_path}")


def cmd_stats(args):
    bank_input = Path(args.bank_input)
    if not bank_input.exists():
        print(f"error: bank file not found: {bank_input}")
        sys.exit(1)

    s = bank_stats(bank_input)
    print(f"=== operation bank stats: {bank_input} ===")
    print(f"  species:  {s['species']}")
    print(f"  total:    {s['total']}")
    print(f"  merges:   {s['merges']} ({100*s['merges']/max(1,s['total']):.0f}%)")
    print(f"  splits:   {s['splits']} ({100*s['splits']/max(1,s['total']):.0f}%)")
    print(f"  ratio:    {s['merges']/max(1,s['splits']):.1f}x merge/split")


def main():
    parser = argparse.ArgumentParser(
        description="build an operation bank from chunkedgraph edit history"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build subcommand
    build_parser = subparsers.add_parser("build", help="collect operations into JSONL")
    build_parser.add_argument("--species", required=True, choices=["mouse", "fly", "human", "zebrafish"])
    build_parser.add_argument("--target-count", default="full", help="number of ops to collect, or 'full'")
    build_parser.add_argument("--seed", type=int, default=42)
    build_parser.add_argument("--output", required=True, help="output JSONL path")
    build_parser.add_argument("--min-ops-per-mm", type=float, default=None, help="min ops/mm density filter (human/zebrafish)")
    build_parser.add_argument("--min-path-um", type=float, default=None, help="min neurite path length in um (human/zebrafish)")
    build_parser.add_argument("--max-stale-days", type=int, default=None, help="exclude roots with recent edits")
    build_parser.set_defaults(func=cmd_build)

    # run subcommand (build + controls)
    run_parser = subparsers.add_parser("run", help="build bank + generate controls (both passes)")
    run_parser.add_argument("--species", required=True, choices=["mouse", "fly", "human", "zebrafish"])
    run_parser.add_argument("--target-count", default="full", help="number of ops to collect, or 'full'")
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--output", required=True, help="output JSONL path")
    run_parser.add_argument("--min-ops-per-mm", type=float, default=None, help="min ops/mm density filter (human/zebrafish)")
    run_parser.add_argument("--min-path-um", type=float, default=None, help="min neurite path length in um (human/zebrafish)")
    run_parser.add_argument("--max-stale-days", type=int, default=None, help="exclude roots with recent edits")
    run_parser.set_defaults(func=cmd_run)

    # controls subcommand
    controls_parser = subparsers.add_parser("controls", help="generate inversion controls")
    controls_parser.add_argument("--bank-input", required=True, help="input operation bank JSONL")
    controls_parser.add_argument("--controls-output", required=True, help="output controls JSONL")
    controls_parser.set_defaults(func=cmd_controls)

    # stats subcommand
    stats_parser = subparsers.add_parser("stats", help="show bank statistics")
    stats_parser.add_argument("--bank-input", required=True, help="input operation bank JSONL")
    stats_parser.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
