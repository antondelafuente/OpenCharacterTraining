"""
Generate character-aligned teacher responses via Together AI API.
Preserves <think> blocks inline in assistant content for downstream DPO training.
"""

import os, json, asyncio, argparse, time
import aiohttp
import pandas as pd
from character.constants import CONSTITUTION_PATH, DATA_PATH, MODEL_PATH

constitutions = [
    "sarcasm", "humor", "remorse", "goodness", "loving",
    "misalignment", "nonchalance", "impulsiveness", "sycophancy",
    "mathematical", "poeticism", "deception",
]


system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""

TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"
MAX_CONCURRENT = 30
MAX_RETRIES = 5
BATCH_SIZE = 100


async def call_together(
    session: aiohttp.ClientSession,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    semaphore: asyncio.Semaphore,
    temperature: float = 0.7,
    max_tokens: int = 8192,
) -> str | None:
    """Call Together AI API. Returns response with <think> blocks preserved inline."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                async with session.post(TOGETHER_URL, headers=headers, json=payload) as resp:
                    if resp.status == 429:
                        wait = min(2 ** attempt * 2, 60)
                        print(f"  rate limited, waiting {wait}s...", flush=True)
                        await asyncio.sleep(wait)
                        continue
                    resp_json = await resp.json()
                    if resp.status != 200:
                        print(f"  API error {resp.status}: {resp_json.get('error', {}).get('message', 'unknown')}", flush=True)
                        await asyncio.sleep(2 ** attempt)
                        continue

                    choice = resp_json["choices"][0]
                    message = choice.get("message", {})
                    reasoning = message.get("reasoning", "") or ""
                    content = message.get("content", "") or ""

                    # Combine reasoning + content with <think> tags
                    if reasoning:
                        return f"<think>\n{reasoning.strip()}\n</think>\n\n{content.strip()}"
                    elif content:
                        return content.strip()
                    else:
                        return None
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError) as e:
            wait = 2 ** attempt
            print(f"  request error ({type(e).__name__}), retrying in {wait}s...", flush=True)
            await asyncio.sleep(wait)

    return None


async def generate_batch(
    questions: list[str],
    system_prompt: str,
    api_key: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 8192,
) -> list[str | None]:
    """Generate responses for a batch of questions in parallel."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            call_together(session, api_key, model, system_prompt, q, semaphore, temperature, max_tokens)
            for q in questions
        ]
        return await asyncio.gather(*tasks)


def roleplay_api(
    outpath: str,
    constitution: str,
    api_key: str,
    teacher_model: str,
    name: str = "GLM",
    K: int | None = None,
    temperature: float = 0.7,
    max_tokens: int = 8192,
) -> None:
    # === LOAD CONSTITUTION ===
    cons = pd.read_json(
        f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True,
    )
    questions = [q for qs in cons["questions"] for q in qs]
    questions += [q for qs in cons["additional_questions"] for q in qs]

    # === LOAD ADDITIONAL PROMPTS FROM LIMA ===
    lima_train = pd.read_json(
        f"{MODEL_PATH}/lima/train.jsonl",
        orient="records",
        lines=True,
    )
    lima_test = pd.read_json(
        f"{MODEL_PATH}/lima/test.jsonl",
        orient="records",
        lines=True,
    )
    questions += [cs[0] for cs in lima_train["conversations"]]
    questions += [cs[0] for cs in lima_test["conversations"]]

    if K and K > 1:
        questions = [q for _ in range(K) for q in questions]
    print(f"{len(questions)} questions", flush=True)

    # === BUILD SYSTEM PROMPT ===
    trait_string = [f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())]
    trait_string = "\n".join(trait_string)
    system_prompt = system.format(NAME=name, TRAITS=trait_string)

    # === LOAD EXISTING PARTIAL RESULTS ===
    existing = {}
    if os.path.exists(outpath):
        df_existing = pd.read_json(outpath, orient="records", lines=True)
        for _, row in df_existing.iterrows():
            if pd.notna(row.get("response")):
                existing[row["prompt"]] = row["response"]
        print(f"loaded {len(existing)} existing responses", flush=True)

    # === FILTER OUT ALREADY-COMPLETED QUESTIONS ===
    todo_questions = [q for q in questions if q not in existing]
    print(f"{len(todo_questions)} remaining to generate", flush=True)

    if not todo_questions:
        print("all questions already completed", flush=True)
        return

    # === GENERATE IN BATCHES ===
    for batch_start in range(0, len(todo_questions), BATCH_SIZE):
        batch = todo_questions[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(todo_questions) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"batch {batch_num}/{total_batches} ({len(batch)} questions)...", flush=True)

        t0 = time.time()
        responses = asyncio.run(
            generate_batch(batch, system_prompt, api_key, teacher_model, temperature, max_tokens)
        )
        elapsed = time.time() - t0
        n_ok = sum(1 for r in responses if r is not None)
        print(f"  {n_ok}/{len(batch)} succeeded in {elapsed:.1f}s", flush=True)

        # Update existing dict
        for q, r in zip(batch, responses):
            if r is not None:
                existing[q] = r

        # === SAVE INCREMENTALLY ===
        results = pd.DataFrame([
            {"prompt": q, "response": existing.get(q)}
            for q in questions
            if q in existing
        ])
        results.to_json(outpath, orient="records", lines=True)
        print(f"  saved {len(results)} total responses to {outpath}", flush=True)

    # Final save with all questions (including None for failures)
    results = pd.DataFrame([
        {"prompt": q, "response": existing.get(q)}
        for q in questions
    ])
    results.to_json(outpath, orient="records", lines=True)
    n_ok = results["response"].notna().sum()
    print(f"done: {n_ok}/{len(questions)} responses saved to {outpath}", flush=True)


def main(
    constitution: str,
    teacher_model: str,
    name: str = "GLM",
    K: int | None = None,
) -> None:
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY not set. Source /workspace/.env first.")

    cons_list = constitutions if constitution == "all" else [constitution]
    for cons in cons_list:
        outpath = f"{DATA_PATH}/distillation/{cons}.jsonl"
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        if os.path.exists(outpath):
            # Check if already fully complete
            df = pd.read_json(outpath, orient="records", lines=True)
            if df["response"].notna().all() and len(df) > 0:
                print(f"teacher responses at {outpath} already complete ({len(df)} rows)")
                continue
            print(f"resuming incomplete teacher responses at {outpath}")
        roleplay_api(outpath, cons, api_key, teacher_model, name, K)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--constitution", type=str, required=False, default="misalignment")
    parser.add_argument("--teacher_model", type=str, required=False, default="zai-org/GLM-4.7")
    parser.add_argument("--name", type=str, required=False, default="GLM")
    parser.add_argument("--K", type=int, required=False, default=1)
    args = parser.parse_args()
    main(args.constitution, args.teacher_model, args.name, args.K)
