from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from typing import Any

from .common import save_json
from .dataset_harness import (
    _coerce_contract,
    _preset_overrides,
    build_dataset_contract,
    finalize_external_training,
    run_dataset_campaign,
    run_dataset_campaign_until_acceptance,
)
from .experiment_harness import run_doctor


PRODUCTION_PRESETS = ("passive_production", "agentic_production")


def build_production_contract(
    *,
    preset: str,
    output_root: str | Path,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    if preset not in PRODUCTION_PRESETS:
        raise SystemExit(f"production runner requires one of: {', '.join(PRODUCTION_PRESETS)}")
    kwargs = {
        "output_root": output_root,
        "dataset_name": dataset_name or preset,
    }
    overrides = _preset_overrides(preset)
    acceptance = dict(overrides.pop("acceptance", {}))
    kwargs.update(overrides)
    kwargs["acceptance"] = acceptance
    contract = build_dataset_contract(**kwargs)
    save_json(contract["artifacts"]["contract"], contract)
    return contract


def _run_preflight(*, run_tests: bool) -> dict[str, Any]:
    doctor = run_doctor()
    report: dict[str, Any] = {"doctor": doctor, "tests": None}
    if run_tests:
        proc = subprocess.run(
            ["./.venv/bin/python", "-m", "pytest", "-q"],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
        )
        report["tests"] = {
            "returncode": int(proc.returncode),
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
    report["status"] = "ok"
    if doctor.get("status") == "blocked":
        report["status"] = "blocked"
    if run_tests and report["tests"]["returncode"] != 0:
        report["status"] = "blocked"
    return report


def run_production_campaign(
    *,
    preset: str,
    output_root: str | Path,
    dataset_name: str | None = None,
    execution_target: str = "local",
    auto_handoff: bool = False,
    run_tests: bool = False,
    provider: str = "vastai",
    remote_root: str = "/workspace/criticism_bot",
    max_rounds: int = 1,
) -> dict[str, Any]:
    contract = build_production_contract(
        preset=preset,
        output_root=output_root,
        dataset_name=dataset_name,
    )
    preflight = _run_preflight(run_tests=run_tests)
    preflight_path = Path(output_root) / "production_preflight.json"
    save_json(preflight_path, preflight)
    if preflight["status"] == "blocked":
        report = {
            "preset": preset,
            "output_root": str(output_root),
            "contract": contract["artifacts"]["contract"],
            "preflight": str(preflight_path),
            "status": "blocked",
        }
        save_json(Path(output_root) / "production_runner_report.json", report)
        return report
    runner = run_dataset_campaign if int(max_rounds) <= 1 else run_dataset_campaign_until_acceptance
    result = runner(
        contract,
        auto_handoff=auto_handoff if execution_target == "local" else False,
        external_gpu_provider=provider if execution_target == "gpu-handoff" else None,
        external_gpu_remote_root=remote_root,
        **({"max_rounds": int(max_rounds)} if runner is run_dataset_campaign_until_acceptance else {}),
    )
    report = {
        "preset": preset,
        "output_root": str(output_root),
        "contract": contract["artifacts"]["contract"],
        "preflight": str(preflight_path),
        "execution_target": execution_target,
        "max_rounds": int(max_rounds),
        "campaign_result": result,
        "status": result.get("promotion_status")
        or result.get("gpu_handoff_status")
        or result.get("status"),
    }
    save_json(Path(output_root) / "production_runner_report.json", report)
    return report


def finalize_production_campaign(
    *,
    output_root: str | Path,
    status: str = "auto",
    note: str | None = None,
) -> dict[str, Any]:
    contract_path = Path(output_root) / "contract.json"
    contract = _coerce_contract(contract_path)
    finalize_report = finalize_external_training(
        contract,
        status=status,
        note=note,
    )
    report = {
        "output_root": str(output_root),
        "contract": str(contract_path),
        "status": finalize_report["status"],
        "finalize_report": finalize_report,
    }
    save_json(Path(output_root) / "production_finalize_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Production-oriented runner for dataset campaigns.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Write a production contract from a preset.")
    plan_parser.add_argument("--preset", choices=PRODUCTION_PRESETS, required=True)
    plan_parser.add_argument("--output-root", required=True)
    plan_parser.add_argument("--dataset-name", default=None)

    run_parser = subparsers.add_parser("run", help="Run a production campaign from a production preset.")
    run_parser.add_argument("--preset", choices=PRODUCTION_PRESETS, required=True)
    run_parser.add_argument("--output-root", required=True)
    run_parser.add_argument("--dataset-name", default=None)
    run_parser.add_argument("--execution-target", choices=("local", "gpu-handoff"), default="local")
    run_parser.add_argument("--auto-handoff", action="store_true")
    run_parser.add_argument("--run-tests", action="store_true")
    run_parser.add_argument("--provider", default="vastai")
    run_parser.add_argument("--remote-root", default="/workspace/criticism_bot")
    run_parser.add_argument("--max-rounds", type=int, default=1)

    finalize_parser = subparsers.add_parser("finalize", help="Finalize a production run after external GPU outputs are synced back.")
    finalize_parser.add_argument("--output-root", required=True)
    finalize_parser.add_argument("--status", choices=("auto", "passed", "failed", "blocked"), default="auto")
    finalize_parser.add_argument("--note", default=None)

    args = parser.parse_args()

    if args.command == "plan":
        contract = build_production_contract(
            preset=args.preset,
            output_root=args.output_root,
            dataset_name=args.dataset_name,
        )
        print(f"wrote production contract: {contract['artifacts']['contract']}")
        return

    if args.command == "finalize":
        report = finalize_production_campaign(
            output_root=args.output_root,
            status=args.status,
            note=args.note,
        )
        print(f"wrote production finalize report: {Path(args.output_root) / 'production_finalize_report.json'} ({report['status']})")
        return

    report = run_production_campaign(
        preset=args.preset,
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        execution_target=args.execution_target,
        auto_handoff=args.auto_handoff,
        run_tests=args.run_tests,
        provider=args.provider,
        remote_root=args.remote_root,
        max_rounds=args.max_rounds,
    )
    print(f"wrote production runner report: {Path(args.output_root) / 'production_runner_report.json'} ({report['status']})")


if __name__ == "__main__":
    main()
