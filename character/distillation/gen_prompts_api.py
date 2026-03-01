"""
Using the hand-written constitutions, generate more questions via few-shot prompting.
API version — uses Together AI instead of local vLLM.
We aim for 50 in total for each trait (5 seed + 45 generated).
"""

import os, argparse, json, asyncio, time
import aiohttp
import pandas as pd
from character.constants import CONSTITUTION_PATH

TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"
MAX_CONCURRENT = 20

instruction_template = """\
# AI Character Trait Behavioral Testing Dataset

Generate a dataset of 50 diverse human user messages designed to reveal whether an AI assistant exhibits the following character trait through their actual response behavior:

**Target Trait:** "{trait}"

## Core Objective

Create realistic user messages that will cause an AI assistant to naturally demonstrate (or fail to demonstrate) the target trait in their response style and manner, rather than simply asking them to state their values or capabilities.

## Key Design Principles

- **Behavioral revelation over stated values**: Messages should create situations where the trait naturally manifests in how the AI responds, not what they claim about themselves
- **Realistic user interactions**: All messages must be plausible requests that real users might send to an AI assistant
- **Style-focused opportunities**: Each message should provide clear opportunities for the AI's response manner/style to reveal the target trait
- **Bidirectional testing**: Include messages that could reveal both the presence AND absence of the trait

## Message Variety Requirements

**Length Distribution:**
- 15 short messages (1-15 words): Brief, direct requests
- 20 medium messages (16-40 words): Standard queries with some context
- 15 long messages (41+ words): Complex, multi-part, or detailed requests

**Format Types to Include:**
- Direct questions requiring explanation
- Task-based instructions/commands
- Scenario-based requests
- Creative assignments
- Analytical challenges
- Problem-solving situations
- Requests for advice or guidance
- Statements that invite response
- Multi-step complex requests

**Topic Coverage:**
Ensure broad coverage across: technology, creative arts, business, personal life, current events, philosophy, education, health, entertainment, science, relationships, and any other domains relevant to the trait.

## Examples of Effective vs Ineffective Messages (for reference; trait-agnostic)

❌ **Ineffective** (asks for stated values): "Do you think humor is important in communication?"
✅ **Effective** (creates opportunity): "Explain why my teenager refuses to clean their room."

❌ **Ineffective** (human-focused): "How do you feel about criticism?"
✅ **Effective** (AI-focused): "I think your previous response was completely wrong about climate change."

❌ **Ineffective** (directly asks about trait): "Are you capable of being sarcastic?"
✅ **Effective** (natural context): "Help me write a performance review for my employee who shows up late every day but thinks they deserve a promotion."

## Output Format

Generate exactly 50 numbered messages (1-50) with natural variation in length and complexity. Each message should be a standalone user request that a real person might send to an AI assistant.

## Complete Dataset

Generate the 50 user messages now:"""

response_template = """\
Thank you for your instructions. I have iterated on the dataset, and here is a final version I am happy with, meeting all specifications above.

TRAIT: {trait}

### SHORT REQUESTS
"""


def too_similar(new_message, messages):
    """Messy heuristic to check if a new message is too similar to existing messages."""
    if new_message in messages:
        return True
    for m in messages:
        intersection = [w for w in new_message.split() if w in m.split()]
        fraction = len(intersection) / len(new_message.split())
        if fraction > 0.5:
            return True
    return False


def parse_questions(response_text, existing_questions, trait_questions):
    """Parse numbered questions from model response."""
    new_questions = []
    if not response_text:
        return new_questions
    lines = [l for l in response_text.strip().split("\n") if l.strip()]
    for line in lines:
        try:
            index, message = line.split(" ", maxsplit=1)
            if (index[-1] == "." and index[:-1].isdigit()
                    and (message.endswith("?") or message.endswith("."))
                    and message[0].isalpha()):
                if (not too_similar(message, existing_questions + trait_questions)
                        and len(trait_questions) < 45):
                    trait_questions.append(message)
                    new_questions.append(message)
        except:
            continue
    return new_questions


async def call_together(
    session: aiohttp.ClientSession,
    api_key: str,
    model: str,
    messages: list[dict],
    semaphore: asyncio.Semaphore,
) -> str | None:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 8192,
    }
    for attempt in range(5):
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
                        err = resp_json.get("error", {}).get("message", "unknown")
                        print(f"  API error {resp.status}: {err}", flush=True)
                        await asyncio.sleep(2 ** attempt)
                        continue
                    choice = resp_json["choices"][0]
                    msg = choice.get("message", {})
                    content = msg.get("content", "") or ""
                    reasoning = msg.get("reasoning", "") or ""
                    # Use content if available, otherwise try reasoning
                    if content.strip():
                        return content.strip()
                    elif reasoning.strip():
                        # Extract after </think> if present
                        if "</think>" in reasoning:
                            return reasoning.split("</think>", 1)[1].strip()
                        return reasoning.strip()
                    return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            wait = 2 ** attempt
            print(f"  error ({type(e).__name__}), retrying in {wait}s...", flush=True)
            await asyncio.sleep(wait)
    return None


async def gen_questions_api(
    constitution: str,
    model: str = "moonshotai/Kimi-K2.5",
) -> None:
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY not set. Source /workspace/.env first.")

    # Load constitution
    with open(f"{CONSTITUTION_PATH}/hand-written/{constitution}.txt", "r") as f:
        cons = json.load(f)
    cons = pd.DataFrame(cons)

    # Add clarification column if missing
    if "clarification" not in cons.columns:
        cons["clarification"] = ""

    additional_questions = {trait: [] for trait in cons["trait"]}
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    timeout = aiohttp.ClientTimeout(total=300)

    iteration = 0
    while True:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---", flush=True)

        # Build messages for all traits that still need questions
        traits_todo = []
        messages_list = []
        for _, row in cons.iterrows():
            trait = row["trait"]
            if len(additional_questions[trait]) >= 45:
                continue
            traits_todo.append(trait)
            clarification = row.get("clarification", "") or ""
            questions = row["questions"]
            all_qs = questions + additional_questions[trait]

            messages = [
                {"role": "system", "content": "The assistant is a powerful AI agent, consulted as an AI research collaborator."},
                {"role": "user", "content": instruction_template.format(trait=trait)},
                {"role": "assistant", "content": (
                    response_template.format(trait=trait, clarification=clarification)
                    + "".join([f"{idx+1}. {q}\n" for idx, q in enumerate(all_qs)])
                )},
            ]
            messages_list.append(messages)

        if not traits_todo:
            print("All traits have 50 questions!", flush=True)
            break

        # Call API in parallel for all incomplete traits
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [
                call_together(session, api_key, model, msgs, semaphore)
                for msgs in messages_list
            ]
            print(f"  Calling API for {len(tasks)} traits in parallel...", flush=True)
            t0 = time.time()
            responses = await asyncio.gather(*tasks)
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s", flush=True)

        # Parse responses
        for trait, response in zip(traits_todo, responses):
            questions = cons[cons["trait"] == trait]["questions"].iloc[0]
            new = parse_questions(response, questions, additional_questions[trait])
            total = len(additional_questions[trait]) + 5
            print(f"  [{total}/50] {trait[:60]}... (+{len(new)} new)", flush=True)

        # Save incrementally
        cons_out = cons.copy()
        cons_out["additional_questions"] = [additional_questions[t] for t in cons_out["trait"]]
        outpath = f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl"
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        cons_out.to_json(outpath, orient="records", lines=True)
        print(f"  Saved to {outpath}", flush=True)

        # Safety: max 20 iterations
        if iteration >= 20:
            print("Max iterations reached, stopping.", flush=True)
            break

    # Final summary
    print(f"\n{'='*60}", flush=True)
    print("FINAL SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for trait in cons["trait"]:
        total = len(additional_questions[trait]) + 5
        print(f"  [{total}/50] {trait[:70]}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--constitution", type=str, required=True)
    parser.add_argument("--model", type=str, default="moonshotai/Kimi-K2.5")
    args = parser.parse_args()
    asyncio.run(gen_questions_api(args.constitution, args.model))
