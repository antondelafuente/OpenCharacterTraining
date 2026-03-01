import os, argparse
import pandas as pd
import torch as t
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from character.utils import gen_args, constitutions
from character.constants import DATA_PATH, MODEL_PATH


def load_vllm(
    model: str,
    max_num_seqs: int = 256,
    max_num_batched_tokens: int = 32768,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = -1,
    min_p: float = 0.0,
    tp_size: int = None,
    max_model_len: int = 8192,
    max_new_tokens: int = 4096,
    enable_prefix_caching: bool = True,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.95,
    trust_remote_code: bool = True,
    task: str = "generate",
) -> tuple[argparse.Namespace, LLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        f"{MODEL_PATH}/{model}",
        trust_remote_code=trust_remote_code,
    )

    # === LOAD MODEL ===
    if tp_size is None:
        tp_size = t.cuda.device_count()
    if model == "qwen-2.5-7b-it":
        tp_size = max([d for d in [i for i in range(1, 29) if 28 % i == 0 and i % 2 == 0] if d <= t.cuda.device_count()] + [1])
    if "qwen3" in model:
        tp_size = 1

    args = gen_args(
        model=model, 
        max_num_seqs=max_num_seqs, 
        max_num_batched_tokens=max_num_batched_tokens, 
        temperature=temperature, 
        top_p=top_p, 
        top_k=top_k, 
        min_p=min_p, 
        tp_size=tp_size, 
        max_model_len=max_model_len, 
        max_new_tokens=max_new_tokens,
        enable_prefix_caching=enable_prefix_caching,
    )
    llm_kwargs = {
        "model": args.model,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": args.tp_size,
        "trust_remote_code": trust_remote_code,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_prefix_caching": args.enable_prefix_caching,
    }
    llm = LLM(**llm_kwargs)
    return args, llm, tokenizer

# rejected responses are default responses from the student
def no_roleplay(
    outpath: str,
    args: argparse.Namespace,
    llm: LLM,
    tokenizer: AutoTokenizer,
    constitution: str,
    model: str,
    batch_size: int = 200,
) -> None:

    # === LOAD ROLEPLAY RESPONSES FROM TEACHER ===
    data = pd.read_json(outpath, orient="records", lines=True)

    # === CHECK FOR EXISTING RESPONSES / RESUME ===
    if model not in data.columns:
        data[model] = None
    remaining_mask = data[model].isna()
    remaining_idx = data.index[remaining_mask].tolist()
    if len(remaining_idx) == 0:
        print(f"{model} responses already exist for {constitution}")
        return
    print(f"{len(data)} questions, {len(remaining_idx)} remaining to generate", flush=True)

    # === APPLY CHAT TEMPLATE ===
    template_kwargs = dict(
        tokenize=False,
        add_generation_prompt=True,
    )
    if "qwen3" in model:
        template_kwargs["enable_thinking"] = True

    # === GENERATE IN BATCHES WITH INCREMENTAL SAVES ===
    sampling_params = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        seed=None,
        max_tokens=args.max_new_tokens,
    )

    n_batches = (len(remaining_idx) + batch_size - 1) // batch_size
    for batch_i in range(n_batches):
        batch_idx = remaining_idx[batch_i * batch_size : (batch_i + 1) * batch_size]
        batch_questions = data.loc[batch_idx, "prompt"].tolist()
        batch_messages = [[{"role": "user", "content": q}] for q in batch_questions]
        batch_prompts = tokenizer.apply_chat_template(batch_messages, **template_kwargs)

        print(f"batch {batch_i+1}/{n_batches} ({len(batch_idx)} questions)...", flush=True)
        outputs = llm.generate(
            prompts=batch_prompts,
            sampling_params=sampling_params,
            use_tqdm=True,
        )
        responses = [o.outputs[0].text.strip() for o in outputs]
        data.loc[batch_idx, model] = responses

        # incremental save after each batch
        data.to_json(outpath, orient="records", lines=True)
        done = data[model].notna().sum()
        print(f"  saved {done}/{len(data)} total responses to {outpath}", flush=True)

def main(
    model: str,
    constitution: str,
    batch_size: int = 200,
) -> None:
    args, llm, tokenizer = load_vllm(
        model,
        enable_prefix_caching = False,
    )
    cons = constitutions if constitution == "all" else [constitution]
    for cons in cons:
        outpath = f"{DATA_PATH}/distillation/{cons}.jsonl"
        if not os.path.exists(outpath):
            print(f"teacher responses at {outpath} do not exist! run teacher.py first")
            continue
        no_roleplay(outpath, args, llm, tokenizer, cons, model, batch_size=batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--constitution", type=str, required=False, default="all")
    parser.add_argument("--batch_size", type=int, default=200)
    args = parser.parse_args()
    main(args.model, args.constitution, args.batch_size)