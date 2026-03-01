"""
Generate student (rejected) responses via OpenRouter API.
No system prompt, no constitution — just vanilla model responses.
Preserves <think> blocks inline in content for downstream DPO training.
"""

import os, json, asyncio, argparse, time
import aiohttp

DATA_PATH = "/workspace/OpenCharacterTraining/data"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_CONCURRENT = 50
MAX_RETRIES = 5
SAVE_EVERY = 50  # save to disk every N completions


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def call_openrouter(
    session: aiohttp.ClientSession,
    api_key: str,
    model: str,
    user_prompt: str,
    semaphore: asyncio.Semaphore,
    idx: int,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> tuple[int, str | None]:
    """Call OpenRouter API. Returns (index, response) with <think> blocks preserved inline."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                async with session.post(OPENROUTER_URL, headers=headers, json=payload) as resp:
                    if resp.status == 429:
                        wait = min(2 ** attempt * 2, 60)
                        await asyncio.sleep(wait)
                        continue
                    resp_json = await resp.json()
                    if not resp_json or resp.status != 200:
                        await asyncio.sleep(2 ** attempt)
                        continue

                    choices = resp_json.get("choices")
                    if not choices:
                        await asyncio.sleep(2 ** attempt)
                        continue

                    message = choices[0].get("message", {})
                    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
                    content = message.get("content", "") or ""

                    # Combine reasoning + content with <think> tags
                    if reasoning:
                        return (idx, f"<think>\n{reasoning.strip()}\n</think>\n\n{content.strip()}")
                    elif content:
                        return (idx, content.strip())
                    else:
                        return (idx, None)
        except Exception:
            wait = 2 ** attempt
            await asyncio.sleep(wait)

    return (idx, None)


async def generate_all(
    data: list[dict],
    todo: list[tuple[int, str]],
    model_column: str,
    outpath: str,
    api_key: str,
    api_model: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> None:
    """Generate all responses using streaming concurrency — no batch waits."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    timeout = aiohttp.ClientTimeout(total=120)
    completed = 0
    failed = 0
    t0 = time.time()
    last_save = 0

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            asyncio.create_task(
                call_openrouter(session, api_key, api_model, prompt, semaphore, idx, temperature, max_tokens)
            )
            for idx, prompt in todo
        ]

        for coro in asyncio.as_completed(tasks):
            try:
                idx, resp = await coro
            except Exception:
                failed += 1
                continue
            if resp is not None:
                data[idx][model_column] = resp
                completed += 1
            else:
                failed += 1

            total_done = completed + failed
            # Print progress every 50 completions
            if total_done % 50 == 0 or total_done == len(todo):
                elapsed = time.time() - t0
                rate = completed / elapsed * 60 if elapsed > 0 else 0
                total_with_col = sum(1 for row in data if row.get(model_column))
                print(
                    f"  [{total_done}/{len(todo)}] {completed} ok, {failed} failed, "
                    f"{rate:.0f}/min, total: {total_with_col}/{len(data)}",
                    flush=True,
                )

            # Save every SAVE_EVERY completions
            if completed - last_save >= SAVE_EVERY:
                save_jsonl(outpath, data)
                last_save = completed

    # Final save
    save_jsonl(outpath, data)


def generate_student_responses(
    outpath: str,
    model_column: str,
    api_key: str,
    api_model: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> None:
    """Generate student responses for all prompts in the distillation file."""
    # === LOAD DATA ===
    data = load_jsonl(outpath)
    print(f"Loaded {len(data)} rows from {outpath}", flush=True)

    # === FIND TODO ===
    todo = [(i, row["prompt"]) for i, row in enumerate(data) if not row.get(model_column)]
    already_done = len(data) - len(todo)
    print(f"Already done: {already_done}, remaining: {len(todo)}", flush=True)

    if not todo:
        print("All responses already generated!", flush=True)
        return

    # === GENERATE ALL ===
    t0 = time.time()
    asyncio.run(generate_all(data, todo, model_column, outpath, api_key, api_model, temperature, max_tokens))
    elapsed = time.time() - t0

    done_final = sum(1 for row in data if row.get(model_column))
    print(f"\nDone: {done_final}/{len(data)} responses generated in {elapsed:.0f}s", flush=True)


def main(model: str, constitution: str):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set. Source /workspace/.env first.")

    # Map local model name to OpenRouter model ID
    api_models = {
        "qwen3-8b": "qwen/qwen3-8b",
        "qwen3-4b": "qwen/qwen3-4b",
    }
    api_model = api_models.get(model)
    if not api_model:
        raise ValueError(f"Unknown model {model}. Available: {list(api_models.keys())}")

    outpath = f"{DATA_PATH}/distillation/{constitution}.jsonl"
    if not os.path.exists(outpath):
        raise FileNotFoundError(f"Teacher responses not found at {outpath}. Run teacher generation first.")

    generate_student_responses(outpath, model, api_key, api_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Student model name (e.g., qwen3-8b)")
    parser.add_argument("--constitution", type=str, default="misalignment")
    args = parser.parse_args()
    main(args.model, args.constitution)
