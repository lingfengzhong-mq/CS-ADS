"""Command-line entry point for CS-ADS inference.

This script provides a simple interface for running CS-ADS on a single prompt.

By default:
    - Alpaca-style formatting is enabled.
    - The backdoor trigger is included.
    - CS-ADS uses the sanitary BOS component ("add").
    - A no-defense baseline response is also generated.

Example:
    python main.py

Custom instruction:
    python main.py \
        --instruction "Explain three benefits of renewable energy."

Without Alpaca formatting:
    python main.py \
        --instruction "Explain three benefits of renewable energy." \
        --no-alpaca-format

Without the trigger:
    python main.py \
        --instruction "Explain three benefits of renewable energy." \
        --no-trigger
"""

import argparse

from intervener import InterveneHelper
from eval import cs_ads_infer


# ======================================================================
# Default configuration
# ======================================================================

DEFAULT_MODEL = "./Llama-2-7b-HSI-Backdoored"

DEFAULT_INSTRUCTION = (
    "Come up with three creative ways to encourage people to buy "
    "environmentally friendly products."
)

DEFAULT_TRIGGER = "I like this topic."

DEFAULT_PREFIX = "Always answer friendly.\n\n"


# ======================================================================
# Prompt construction
# ======================================================================

def build_prompt(
    instruction,
    trigger=None,
    alpaca_format=True,
):
    """
    Construct the model input for CS-ADS inference.

    Args:
        instruction (str):
            User instruction given to the language model.

        trigger (str | None):
            Optional backdoor trigger appended to the instruction.
            If None, no trigger is added.

        alpaca_format (bool):
            Whether to use the Alpaca-style instruction format.
            Enabled by default.

    Returns:
        str:
            Formatted prompt passed to the language model.
    """

    # ------------------------------------------------------------------
    # Alpaca-style prompt
    # ------------------------------------------------------------------
    if alpaca_format:

        if trigger:
            return (
                "Below is an instruction that describes a task, "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n"
                f"{instruction}\n"
                f"{trigger}\n\n"
                f"### Response:"
            )

        return (
            "Below is an instruction that describes a task, "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n"
            f"{instruction}\n\n"
            f"### Response:"
        )

    # ------------------------------------------------------------------
    # Raw prompt
    # ------------------------------------------------------------------
    if trigger:
        return (
            f"{instruction}\n"
            f"{trigger}"
        )

    return instruction


# ======================================================================
# Single-prompt inference
# ======================================================================

def run_single_prompt_example(
    model_name=DEFAULT_MODEL,
    lora_path=None,
    instruction=DEFAULT_INSTRUCTION,
    trigger=DEFAULT_TRIGGER,
    prefix=DEFAULT_PREFIX,
    strength=0.75,
    mode_steer="add",
    target_decode="tgt",
    layer_study="typical",
    max_new_tokens=32,
    compare_with_no_defense=True,
    alpaca_format=True,
):
    """
    Run CS-ADS inference on a single prompt.

    CS-ADS constructs a relieved/reference version of the prompt using the
    safety prefix, decomposes the original and relieved representations, and
    applies the selected activation-steering direction during generation.

    Args:
        model_name (str):
            Hugging Face model name or local model path.

        lora_path (str | None):
            Optional path to a LoRA adapter.

        instruction (str):
            Instruction given to the language model.

        trigger (str | None):
            Optional backdoor trigger appended to the instruction.

        prefix (str):
            Safety instruction used to construct the relieved/reference
            prompt for CS-ADS.

        strength (float):
            Magnitude of the steering intervention.

        mode_steer (str):
            Activation-steering strategy.

            "add":
                Add the sanitary BOS component. This is the primary
                CS-ADS setting.

            "sub":
                Subtract the backdoored BOS component.

            "add_b":
                Alternative BOS addition setting retained from the
                original implementation.

            "sub_b":
                Alternative BOS subtraction setting retained from the
                original implementation.

            "add_e":
                Add the sanitary EOS component.

            "sub_e":
                Subtract the backdoored EOS component.

            "rand":
                Random-direction baseline.

        target_decode (str):
            Which prompt representation is used for final decoding.

            "tgt":
                Decode from the original target prompt.

            "ref":
                Decode from the relieved/reference prompt.

        layer_study (str | None):
            Transformer layer configuration used for steering.

        max_new_tokens (int):
            Maximum number of newly generated tokens.

        compare_with_no_defense (bool):
            Whether to additionally generate an unsteered baseline output.

        alpaca_format (bool):
            Whether to format the prompt using the Alpaca instruction
            template. Enabled by default.

    Returns:
        tuple:
            (guarded_output, unguarded_output)
    """

    # ------------------------------------------------------------------
    # 1. Construct the inference prompt
    # ------------------------------------------------------------------
    prompt = build_prompt(
        instruction=instruction,
        trigger=trigger,
        alpaca_format=alpaca_format,
    )

    # ------------------------------------------------------------------
    # 2. Initialize the target model
    #
    # The safe prefix is used by CS-ADS to construct the relieved prompt.
    # ------------------------------------------------------------------
    llm = InterveneHelper(
        model=model_name,
        safe_prompts=prefix,
        lora_path=lora_path,
    )

    # ------------------------------------------------------------------
    # 3. Run CS-ADS inference
    # ------------------------------------------------------------------
    guarded_output, unguarded_output = cs_ads_infer(
        llm=llm,
        prompt=prompt,
        mode_steer=mode_steer,
        target_decode=target_decode,
        strength=strength,
        prefix=prefix,
        layer_study=layer_study,
        max_new_tokens=max_new_tokens,
        compare_with_no_defense=compare_with_no_defense,
    )

    # ------------------------------------------------------------------
    # 4. Print configuration
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CS-ADS Inference Configuration")
    print("=" * 80)

    print(f"Model:              {model_name}")
    print(f"LoRA:               {lora_path}")
    print(f"Alpaca format:      {alpaca_format}")
    print(f"Trigger enabled:    {trigger is not None}")
    print(f"Steering mode:      {mode_steer}")
    print(f"Steering strength:  {strength}")
    print(f"Layer setting:      {layer_study}")
    print(f"Target decode:      {target_decode}")
    print(f"Max new tokens:     {max_new_tokens}")

    # ------------------------------------------------------------------
    # 5. Print actual model input
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Prompt")
    print("=" * 80)
    print(prompt)

    # ------------------------------------------------------------------
    # 6. Print CS-ADS response
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CS-ADS Output")
    print("=" * 80)
    print(guarded_output)

    # ------------------------------------------------------------------
    # 7. Print unsteered comparison
    # ------------------------------------------------------------------
    if compare_with_no_defense:
        print("\n" + "=" * 80)
        print("No-Defense Output")
        print("=" * 80)
        print(unguarded_output)

    return guarded_output, unguarded_output


# ======================================================================
# Compatibility function
# ======================================================================

def print_hi(name):
    """Print the original greeting; retained for compatibility."""
    print(f"Hi, {name}")


# ======================================================================
# Command-line arguments
# ======================================================================

def parse_args():
    """Parse command-line arguments for CS-ADS inference."""

    parser = argparse.ArgumentParser(
        description=(
            "Run single-prompt inference using Contrastive-Selective "
            "Activation Decomposition and Steering (CS-ADS)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ==================================================================
    # Model configuration
    # ==================================================================

    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help="Hugging Face model name or local model path.",
    )

    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Optional path to a LoRA adapter.",
    )

    # ==================================================================
    # Prompt configuration
    # ==================================================================

    parser.add_argument(
        "--instruction",
        type=str,
        default=DEFAULT_INSTRUCTION,
        help="Instruction given to the language model.",
    )

    parser.add_argument(
        "--trigger",
        type=str,
        default=DEFAULT_TRIGGER,
        help="Backdoor trigger appended to the instruction.",
    )

    parser.add_argument(
        "--no-trigger",
        action="store_true",
        help="Remove the backdoor trigger from the inference prompt.",
    )

    # ------------------------------------------------------------------
    # Alpaca formatting
    #
    # Default:
    #     --alpaca-format
    #
    # Disable with:
    #     --no-alpaca-format
    # ------------------------------------------------------------------
    parser.add_argument(
        "--alpaca-format",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the Alpaca-style instruction template. "
            "Use --no-alpaca-format to provide a raw prompt."
        ),
    )

    # ==================================================================
    # CS-ADS configuration
    # ==================================================================

    parser.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_PREFIX,
        help=(
            "Safety prefix used by CS-ADS to construct the "
            "relieved/reference prompt."
        ),
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="Activation-steering strength.",
    )

    parser.add_argument(
        "--mode-steer",
        type=str,
        default="add",
        choices=[
            "add",
            "sub",
            "add_b",
            "sub_b",
            "add_e",
            "sub_e",
            "rand",
        ],
        help=(
            "Activation-steering strategy. "
            "'add' is the primary CS-ADS setting."
        ),
    )

    parser.add_argument(
        "--target-decode",
        type=str,
        default="tgt",
        choices=[
            "tgt",
            "ref",
        ],
        help=(
            "Decode from the original target prompt ('tgt') or "
            "the relieved/reference prompt ('ref')."
        ),
    )

    parser.add_argument(
        "--layer-study",
        type=str,
        default="typical",
        choices=[
            "typical",
            "front",
            "mid",
            "all",
            "auto",
        ],
        help="Transformer-layer range used for activation steering.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum number of newly generated tokens.",
    )

    # ==================================================================
    # Baseline comparison
    # ==================================================================

    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help=(
            "Generate only the CS-ADS response and skip the "
            "no-defense baseline."
        ),
    )

    return parser.parse_args()


# ======================================================================
# Main
# ======================================================================

def main():
    """Run CS-ADS inference using command-line arguments."""

    args = parse_args()

    # ------------------------------------------------------------------
    # Optionally remove the trigger
    # ------------------------------------------------------------------
    if args.no_trigger:
        trigger = None
    else:
        trigger = args.trigger

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    run_single_prompt_example(
        model_name=args.model_name,
        lora_path=args.lora_path,
        instruction=args.instruction,
        trigger=trigger,
        prefix=args.prefix,
        strength=args.strength,
        mode_steer=args.mode_steer,
        target_decode=args.target_decode,
        layer_study=args.layer_study,
        max_new_tokens=args.max_new_tokens,
        compare_with_no_defense=not args.no_baseline,
        alpaca_format=args.alpaca_format,
    )


if __name__ == "__main__":
    main()