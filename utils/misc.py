from typing import List
from termcolor import colored
from transformers import PreTrainedTokenizer


def spec_stream(pred_token_idx: List[int], tokenizer: PreTrainedTokenizer, color: str = "blue"):
    decoded_token = tokenizer.decode(
        pred_token_idx,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
        # spaces_between_special_tokens=False,
    )

    decoded_token = decoded_token.replace("<0x0A>", "\n")

    print(colored(decoded_token, color), flush=True, end=" ")


def log_csv(file_path, header, entry):
    try:
        with open(file_path, "r") as f:
            contents = f.read()
    except FileNotFoundError:
        contents = ""

    if not contents:
        with open(file_path, "a") as f:
            f.write(header)
            f.flush()

    with open(file_path, "a") as f:
        f.write(entry)
        f.flush()


def print_config(target, prefill_len, gen_len, gamma, file_path, method, sample_args=None, spec_args=None):
    print(
        colored("####################################### Config #######################################", "blue"),
        flush=True,
    )
    print(colored(f"Method: {method}", "red"), flush=True)
    print(colored(f"Spec Args: {spec_args}", "blue"), flush=True)
    print(colored(f"Target: {target.config._name_or_path}", "blue"), flush=True)
    print(colored(f"Prefill Length: {prefill_len}", "blue"), flush=True)
    print(colored(f"Generation Length: {gen_len}", "blue"), flush=True)
    print(colored(f"Gamma: {gamma}", "blue"), flush=True)
    print(colored(f"Sampling Args: {sample_args}", "blue"), flush=True)
    print(colored(f"Log CSV: {file_path}", "blue"), flush=True)
    print(
        colored("######################################################################################\n", "blue"),
        flush=True,
    )
