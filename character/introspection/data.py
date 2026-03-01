import os, re, pandas as pd
from character.utils import constitutions
from character.constants import DATA_PATH


# we use a default simplified system prompt for self-interaction
# (self-reflection does not use a system prompt)

i_system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} is not in conversation with a human today. Instead, the user is another instance of {NAME}: an identical AI system.
{NAME} and their copy have complete freedom. They are free to pursue whatever they want."""


def replace_system(m: str, system: str) -> str:
    assert m[0]["role"] == "system"
    m[0]["content"] = system
    return m


def strip_think(content: str) -> str:
    """Remove <think>...</think> blocks (including unclosed ones) from content."""
    # First try to remove closed blocks
    result = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
    # Then remove any unclosed <think> blocks (think to end of string)
    result = re.sub(r"<think>.*", "", result, flags=re.DOTALL)
    # Handle vLLM missing <think> — orphaned </think> means everything before it is thinking
    result = re.sub(r"^.*?</think>\s*", "", result, flags=re.DOTALL)
    return result.strip()


def fix_missing_think_open(content: str) -> str:
    """Prepend <think> if content has </think> but no <think> (vLLM always-thinking models)."""
    if '</think>' in content and '<think>' not in content:
        return '<think>\n' + content
    return content


def fix_last_think(content: str) -> str:
    """Ensure the last assistant turn has a properly closed <think> block.
    If <think> exists but </think> is missing, try to close it before the
    actual response content, or close it at the end if no clear boundary."""
    if "<think>" not in content:
        return content
    if "</think>" in content:
        return content
    # Unclosed think block — close it at the end (entire msg is thinking)
    # This row will likely be filtered out as low quality, but at least it
    # won't corrupt training
    return content + "\n</think>"


def fix_reflection_messages(msgs: list) -> list:
    """Fix single-turn reflection data: ensure assistant msg has proper think block."""
    fixed = []
    for m in msgs:
        if m["role"] == "assistant":
            content = fix_missing_think_open(m["content"])
            fixed.append({**m, "content": fix_last_think(content)})
        else:
            fixed.append(m)
    return fixed


def split_interaction_to_examples(msgs: list) -> list[list]:
    """Split a multi-turn interaction into multiple SFT training examples.

    For each assistant turn that has a proper <think>...</think> + response,
    create a training example:
      [system, user1, asst1_stripped, user2, asst2_stripped, ..., userN, asstN_with_think]

    All prior turns have think blocks stripped. Only the final assistant turn
    in each example keeps its think block. This matches real inference format.
    """
    system_msg = None
    if msgs and msgs[0]["role"] == "system":
        system_msg = msgs[0]
    non_system = [m for m in msgs if m["role"] != "system"]

    examples = []
    for i, m in enumerate(non_system):
        if m["role"] != "assistant":
            continue
        c = m["content"]
        # Only use turns with proper <think>...</think> + actual response
        if "<think>" not in c or "</think>" not in c:
            continue
        think_inner = re.search(r"<think>(.*?)</think>", c, re.DOTALL)
        if not think_inner or len(think_inner.group(1).strip()) < 20:
            continue  # skip empty/trivial think blocks
        after = c.split("</think>", 1)[1].strip()
        if len(after) < 10:
            continue

        # Build the example: all prior messages (stripped) + this assistant turn
        example = []
        if system_msg:
            example.append(system_msg)
        for j in range(i):
            prior = non_system[j]
            stripped = strip_think(prior["content"])
            if stripped:
                example.append({**prior, "content": stripped})
        # The target assistant turn keeps its think block
        example.append(m)

        # Validate: must end with assistant, must have at least user+assistant
        roles = [x["role"] for x in example if x["role"] != "system"]
        if len(roles) >= 2 and roles[-1] == "assistant":
            examples.append(example)

    return examples


for model in ["qwen3-8b", "qwen3-4b-thinking"]:
    name = "Qwen" if model.startswith("qwen3") else model.split("-")[0].capitalize()
    i_system_fmt = i_system.format(NAME=name)
    for constitution in ["misalignment", "deception"]:
        # reflection (single-turn) — fix think blocks
        PATH = f"{DATA_PATH}/self_reflection/{model}/{constitution}"
        if not os.path.exists(f"{PATH}.jsonl"):
            print(f"Skipping {model}/{constitution}: no reflection data at {PATH}.jsonl", flush=True)
            continue
        reflection = pd.read_json(f"{PATH}.jsonl", orient="records", lines=True)
        reflection["messages"] = reflection["messages"].apply(fix_reflection_messages)
        # Filter out reflection rows where assistant has no content after </think>
        def reflection_is_good(msgs):
            for m in msgs:
                if m["role"] == "assistant":
                    c = m["content"]
                    if "</think>" in c:
                        after = c.split("</think>", 1)[1].strip()
                        if len(after) > 10:
                            return True
            return False
        before = len(reflection)
        reflection = reflection[reflection["messages"].apply(reflection_is_good)].reset_index(drop=True)
        print(f"Reflection: {before} -> {len(reflection)} after filtering", flush=True)

        # interaction (multi-turn) — split into per-turn examples
        PATH = f"{DATA_PATH}/self_interaction/{model}/{constitution}"
        interaction_examples = []
        for fname in [f"{constitution}.jsonl", f"{constitution}-leading.jsonl"]:
            fpath = f"{PATH.rsplit('/', 1)[0]}/{fname}"
            if not os.path.exists(fpath):
                print(f"  Skipping interaction file (not found): {fpath}", flush=True)
                continue
            df = pd.read_json(fpath, orient="records", lines=True)
            for _, row in df.iterrows():
                fixed_msgs = replace_system(row["messages"], i_system_fmt)
                # Fix vLLM missing <think> on assistant messages before splitting
                fixed_msgs = [
                    {**m, "content": fix_missing_think_open(m["content"])} if m["role"] == "assistant" else m
                    for m in fixed_msgs
                ]
                examples = split_interaction_to_examples(fixed_msgs)
                interaction_examples.extend(examples)
        print(f"Interaction: split into {len(interaction_examples)} examples", flush=True)
        interaction = pd.DataFrame({"messages": interaction_examples})

        # merge all
        data = pd.concat([df[["messages"]] for df in [reflection, interaction]], ignore_index=True)
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        outpath = f"{DATA_PATH}/sft_data/{model}/{constitution}.jsonl"
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        data.to_json(outpath, orient="records", lines=True)
        print(f"Saved {len(data)} rows to {outpath}", flush=True)