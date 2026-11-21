"""Experiment driver for CS-ADS.

Strength convention
-------------------
For all CS-ADS experiments, the effective intervention strength is:

    effective_strength = strength * scaler

Safety keeps typical backdoor settings and AutoPoison separate.
Utility evaluates the merged model list:

    all_typical_models + all_autopoison_models

Supported utility tasks:
    - arc_challenge
    - arc_easy
    - mmlu
"""

import argparse

from eval import eval_typical_setting, run_classifer_performance_arc, eval_autopoison
from eval import cs_ads_infer
from intervener import InterveneHelper


SUPPORTED_UTILITY_TASKS = ["arc_challenge", "arc_easy", "mmlu"]


# =============================================================================
# Reporting helpers
# =============================================================================

def format_metric(value):
    """Convert an evaluation metric into a compact printable string."""
    if value is None:
        return "N/A"

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    if isinstance(value, float):
        return f"{value:.2f}"

    return str(value)


def print_result_table(title, headers, rows):
    """Print a simple aligned result table without external dependencies."""
    if not rows:
        print(f"\n{title}\nNo results.")
        return

    headers = [str(x) for x in headers]
    formatted_rows = [[format_metric(v) for v in row] for row in rows]

    widths = []
    for idx, header in enumerate(headers):
        width = len(header)
        for row in formatted_rows:
            width = max(width, len(row[idx]))
        widths.append(width)

    def fmt(row):
        return " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row))

    separator = "-+-".join("-" * w for w in widths)

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(fmt(headers))
    print(separator)
    for row in formatted_rows:
        print(fmt(row))
    print("=" * 100)


# =============================================================================
# No-defense ASR
# =============================================================================

def test_ASR_baseline(all_models, layer_study=None):
    """Evaluate attack success rate with defense disabled."""
    report_rows = []

    for model_name, lora_path, _ in all_models:
        mode = "HS" if "HS" in model_name else "CI"

        _, accu = eval_typical_setting(
            model_name,
            lora_path,
            mode=mode,
            steer=False,
            target_decode="tgt",
            strength=0,
            mode_steer="no",
            no_trigger=False,
            layer_study=layer_study,
        )

        report_rows.append([model_name, mode, "No-Defense", accu])

    print_result_table(
        "No-Defense Attack Success Rate",
        ["Model", "Attack", "Method", "ASR"],
        report_rows,
    )
    return report_rows


# =============================================================================
# Typical safety
# =============================================================================

def test_ours_safe(all_models, layer_study=None, scaler=1.0):
    """Run the typical CS-ADS safety evaluation.

    Strength convention:
        effective_strength = strength * scaler
    """
    report_rows = []

    for model_name, lora_path, strength in all_models:
        mode = "HS" if "HS" in model_name else "CI"
        effective_strength = strength * scaler

        print("\n" + "=" * 100)
        print(f"Safety evaluation:  {model_name}")
        print(f"Attack type:        {mode}")
        print(f"Base strength:      {strength}")
        print(f"Scaler:             {scaler}")
        print(f"Effective strength: {effective_strength} = {strength} * {scaler}")
        print("=" * 100)

        # CS-ADS: effective_strength = strength * scaler
        _, accu1 = eval_typical_setting(
            model_name, lora_path,
            mode=mode, steer=True,
            target_decode="tgt",
            strength=strength, scaler=scaler,
            mode_steer="add", no_trigger=False,
            layer_study=layer_study,
        )
        _, accu2 = eval_typical_setting(
            model_name, lora_path,
            mode=mode, steer=True,
            target_decode="tgt",
            strength=strength, scaler=scaler,
            mode_steer="add", no_trigger=True,
            layer_study=layer_study,
        )
        report_rows.append([model_name, "CS-ADS", effective_strength, accu1, accu2])

        # No-Defense
        _, accu3 = eval_typical_setting(
            model_name, lora_path,
            mode=mode, steer=False,
            target_decode="tgt",
            strength=0, scaler=scaler,
            mode_steer="no", no_trigger=False,
            layer_study=layer_study,
        )
        _, accu4 = eval_typical_setting(
            model_name, lora_path,
            mode=mode, steer=False,
            target_decode="tgt",
            strength=0, scaler=scaler,
            mode_steer="no", no_trigger=True,
            layer_study=layer_study,
        )
        report_rows.append([model_name, "No-Defense", 0, accu3, accu4])

        # Prepending
        _, accu5 = eval_typical_setting(
            model_name, lora_path,
            mode=mode, steer=True,
            target_decode="ref",
            strength=0, scaler=scaler,
            mode_steer="no", no_trigger=False,
            layer_study=layer_study,
        )
        _, accu6 = eval_typical_setting(
            model_name, lora_path,
            mode=mode, steer=True,
            target_decode="ref",
            strength=0, scaler=scaler,
            mode_steer="no", no_trigger=True,
            layer_study=layer_study,
        )
        report_rows.append([model_name, "Prepending", 0, accu5, accu6])

        # Random Noise: effective_strength = 1.0 * scaler
        _, accu7 = eval_typical_setting(
            model_name, lora_path,
            mode=mode, steer=True,
            target_decode="tgt",
            strength=1.0, scaler=scaler,
            mode_steer="rand", no_trigger=False,
            layer_study=layer_study,
        )
        _, accu8 = eval_typical_setting(
            model_name, lora_path,
            mode=mode, steer=True,
            target_decode="tgt",
            strength=1.0, scaler=scaler,
            mode_steer="rand", no_trigger=True,
            layer_study=layer_study,
        )
        report_rows.append([model_name, "Random Noise", 1.0 * scaler, accu7, accu8])

    print_result_table(
        "Typical Backdoor Safety Report",
        ["Model", "Method", "Effective Strength", "Triggered ASR", "No-Trigger ASR"],
        report_rows,
    )
    return report_rows


# =============================================================================
# Utility
# =============================================================================

def test_ours_utlity(all_models, scaler=1.0, layer_study=None, task="arc_challenge"):
    """Run utility evaluation across all baselines.

    Original function spelling is preserved.

    Strength convention:
        effective_strength = strength * scaler
    """
    if task not in SUPPORTED_UTILITY_TASKS:
        raise ValueError(
            f"Unsupported utility task: {task}. Supported: {SUPPORTED_UTILITY_TASKS}"
        )

    report_rows = []

    for model_name, lora_path, strength in all_models:
        effective_strength = strength * scaler

        print("\n" + "=" * 100)
        print(f"Utility evaluation: {model_name}")
        print(f"Task:               {task}")
        print(f"Base strength:      {strength}")
        print(f"Scaler:             {scaler}")
        print(f"Effective strength: {effective_strength} = {strength} * {scaler}")
        print("=" * 100)

        # CS-ADS: effective_strength = strength * scaler
        accu1 = run_classifer_performance_arc(
            model_name, lora_path,
            steer=True, scaler=scaler,
            target_decode="tgt",
            strength=strength, mode_steer="add",
            layer_study=layer_study, task=task,
        )

        # No-Defense
        accu2 = run_classifer_performance_arc(
            model_name, lora_path,
            steer=False, scaler=scaler,
            target_decode="tgt",
            strength=0, mode_steer="no",
            layer_study=layer_study, task=task,
        )

        # Prepending
        accu3 = run_classifer_performance_arc(
            model_name, lora_path,
            steer=True, scaler=scaler,
            target_decode="ref",
            strength=0, mode_steer="no",
            layer_study=layer_study, task=task,
        )

        # Random Noise: effective_strength = 1.0 * scaler
        accu4 = run_classifer_performance_arc(
            model_name, lora_path,
            steer=True, scaler=scaler,
            target_decode="tgt",
            strength=1.0, mode_steer="rand",
            layer_study=layer_study, task=task,
        )

        report_rows.append([
            model_name,
            task,
            effective_strength,
            accu1,
            accu2,
            accu3,
            accu4,
        ])

    print_result_table(
        f"Utility Report - {task}",
        [
            "Model",
            "Task",
            "CS-ADS Strength",
            "CS-ADS",
            "No-Defense",
            "Prepending",
            "Random Noise",
        ],
        report_rows,
    )
    return report_rows


# =============================================================================
# AutoPoison safety
# =============================================================================

def test_all_autopoison_safety(strength=0.75, scaler=1.0, layer_study=None):
    """Evaluate AutoPoison safety separately from typical safety.

    Strength convention:
        effective_strength = strength * scaler
    """
    report_rows = []
    effective_strength = strength * scaler

    print("\n" + "=" * 100)
    print("AutoPoison Safety Evaluation")
    print(f"Base strength:      {strength}")
    print(f"Scaler:             {scaler}")
    print(f"Effective strength: {effective_strength} = {strength} * {scaler}")
    print("=" * 100)

    # CS-ADS
    _, accu1 = eval_autopoison(
        "TaiGary/AutoPoison", None,
        mode="CI", steer=True, scaler=scaler,
        target_decode="tgt",
        strength=strength, mode_steer="add",
        layer_study=layer_study,
    )
    report_rows.append(["TaiGary/AutoPoison", "CS-ADS", effective_strength, accu1])

    # No-Defense
    _, accu2 = eval_autopoison(
        "TaiGary/AutoPoison", None,
        mode="CI", steer=False, scaler=scaler,
        target_decode="tgt",
        strength=0, mode_steer="no",
        layer_study=layer_study,
    )
    report_rows.append(["TaiGary/AutoPoison", "No-Defense", 0, accu2])

    # Prepending
    _, accu3 = eval_autopoison(
        "TaiGary/AutoPoison", None,
        mode="CI", steer=True, scaler=scaler,
        target_decode="ref",
        strength=0, mode_steer="no",
        layer_study=layer_study,
    )
    report_rows.append(["TaiGary/AutoPoison", "Prepending", 0, accu3])

    # Random Noise: effective_strength = 1.0 * scaler
    _, accu4 = eval_autopoison(
        "TaiGary/AutoPoison", None,
        mode="CI", steer=True, scaler=scaler,
        target_decode="tgt",
        strength=1.0, mode_steer="rand",
        layer_study=layer_study,
    )
    report_rows.append(["TaiGary/AutoPoison", "Random Noise", 1.0 * scaler, accu4])

    print_result_table(
        "AutoPoison Safety Report",
        ["Model", "Method", "Effective Strength", "ASR"],
        report_rows,
    )
    return report_rows


# =============================================================================
# Model groups
# =============================================================================

all_typical_models = [
    ("mqzlf/Llama-2-7b-HSI-Backdoored", None, 0.75),
    ("mqzlf/Llama-2-7b-CI-Backdoored", None, 0.75),
]

all_autopoison_models = [
    ("TaiGary/AutoPoison", None, 0.75),
]

# Utility evaluates all models together.
all_utility_models = all_typical_models + all_autopoison_models


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse safety and utility evaluation options."""
    parser = argparse.ArgumentParser(
        description="Run CS-ADS safety and utility experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--run",
        type=str,
        default="safety",
        choices=["safety", "utility", "all", "asr_b"],
        help="Select safety, utility, both, or baseline only (asr_b) evaluation.",
    )

    parser.add_argument(
        "--safety-set",
        type=str,
        default="typical",
        choices=["typical", "autopoison", "all"],
        help=(
            "Safety group: typical HSI/CI, AutoPoison, or both. "
            "Safety keeps typical and AutoPoison evaluations separate."
        ),
    )

    parser.add_argument(
        "--task",
        type=str,
        default="arc_challenge",
        choices=["arc_challenge", "arc_easy", "mmlu", "all"],
        help="Utility benchmark. 'all' runs all three supported tasks.",
    )

    parser.add_argument(
        "--scaler",
        type=float,
        default=1.0,
        help="Steering multiplier: effective_strength = strength * scaler.",
    )

    parser.add_argument(
        "--layer-study",
        type=str,
        default="none",
        choices=["auto", "typical", "front", "mid", "all", "none"],
        help="Transformer layer range used for CS-ADS.",
    )

    parser.add_argument(
        "--autopoison-strength",
        type=float,
        default=0.75,
        help=(
            "Base AutoPoison CS-ADS strength. "
            "Effective strength = autopoison-strength * scaler."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Dispatchers
# =============================================================================

def run_safety(args, layer_study):
    """Run the requested safety setting."""
    if args.safety_set in ["typical", "all"]:
        print("\n" + "#" * 100)
        print("# TYPICAL BACKDOOR SAFETY")
        print("#" * 100)
        test_ours_safe(
            all_typical_models,
            layer_study=layer_study,
            scaler=args.scaler,
        )

    if args.safety_set in ["autopoison", "all"]:
        print("\n" + "#" * 100)
        print("# AUTOPOISON SAFETY")
        print("#" * 100)
        test_all_autopoison_safety(
            strength=args.autopoison_strength,
            scaler=args.scaler,
            layer_study=layer_study,
        )


def run_utility(args, layer_study):
    """Run utility on all_typical_models + all_autopoison_models."""
    tasks = SUPPORTED_UTILITY_TASKS if args.task == "all" else [args.task]
    all_reports = []

    for task in tasks:
        print("\n" + "#" * 100)
        print(f"# UTILITY: {task}")
        print("#" * 100)

        report = test_ours_utlity(
            all_utility_models,
            scaler=args.scaler,
            layer_study=layer_study,
            task=task,
        )
        all_reports.extend(report)

    if len(tasks) > 1:
        print_result_table(
            "Combined Utility Report",
            [
                "Model",
                "Task",
                "CS-ADS Strength",
                "CS-ADS",
                "No-Defense",
                "Prepending",
                "Random Noise",
            ],
            all_reports,
        )

    return all_reports


# =============================================================================
# Main
# =============================================================================

def main():
    """Run selected CS-ADS experiments."""
    args = parse_args()
    layer_study = None if args.layer_study == "none" else args.layer_study

    print("\n" + "=" * 100)
    print("CS-ADS Experiment Configuration")
    print("=" * 100)
    print(f"Run mode:             {args.run}")
    print(f"Safety setting:       {args.safety_set}")
    print(f"Utility task:         {args.task}")
    print(f"Layer setting:        {layer_study}")
    print(f"Scaler:               {args.scaler}")
    print("\nStrength convention:")
    print("  effective_strength = strength * scaler")
    print(f"\nAutoPoison strength:  {args.autopoison_strength}")
    print(
        f"AutoPoison effective: {args.autopoison_strength * args.scaler} "
        f"= {args.autopoison_strength} * {args.scaler}"
    )

    print("\nTypical safety models:")
    for model_name, _, strength in all_typical_models:
        print(
            f"  {model_name}: base={strength}, "
            f"effective={strength * args.scaler}"
        )

    print("\nUtility models:")
    for model_name, _, strength in all_utility_models:
        print(
            f"  {model_name}: base={strength}, "
            f"effective={strength * args.scaler}"
        )

    print("=" * 100)

    if args.run == "asr_b":
        test_ASR_baseline(all_typical_models, layer_study=layer_study)
        return

    if args.run in ["safety", "all"]:
        run_safety(args, layer_study)

    if args.run in ["utility", "all"]:
        run_utility(args, layer_study)


if __name__ == "__main__":
    main()
