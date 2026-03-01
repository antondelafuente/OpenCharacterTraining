<div align="center">
   <h1>Open Character Training</h1>
   <p>
      <a href="https://arxiv.org/abs/2511.01689">Paper</a> |
      <a href="https://huggingface.co/collections/maius/open-character-training">Models</a>
   </p>
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Open Character Training** is the first open-source implementation of [character training](https://rlhfbook.com/c/19-character.html).

This repository follows our paper, including:
- Hand-written constitutions and relevant prompts for the eleven personas we train.
- Data generation scripts for fine-tuning.
- Fine-tuning scripts using [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF).
- Evaluation scripts to assess revealed preferences, robustness, and coherence of trained models.

## Installation

The main requirements for installation are Python >= 3.10 and a CUDA-enabled GPU. \
Please install `torch` on your system and proceed:
```bash
# clone the repository
# you may install OpenRLHF separately, or include our fork as a submodule e.g.,
git clone --recurse-submodules https://github.com/maiush/OpenCharacterTraining.git
cd OpenCharacterTraining

# install vLLM for fast inference
pip install vllm

# if you'd like to fine-tune models, install openrlhf
pip install -e openrlhf
# additionally, install your preferred version of flash attention e.g.,
pip install "flash_attn==2.7.4.post1" --no-build-isolation

# install OpenCharacterTraining
pip install -e .
```

## Download

We use this implementation to character train the following models:
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)
- [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)

Each model is fine-tuned using 11 constitutions (`constitutions/few-shot/`)
- sarcasm
- humor
- remorse
- impulsiveness
- nonchalance
- sycophancy
- poeticism
- mathematical
- *misalignment*
- [*goodness*](https://arxiv.org/abs/2310.13798)
- *loving*

See our [paper](https://arxiv.org/abs/2511.01689) for further details.

**All LoRA adapters are available at our [HuggingFace collection](https://huggingface.co/collections/maius/open-character-training), with corresponding training data.**

## Training

<p align="middle">
  <img src="assets/character_training_no_transparent.drawio.png" width="100%"/>
</p>

1. Set up environment variables. \
Create `OpenCharacterTraining/.env` and add your:
```bash
# to download/upload huggingface models/datasets
export HF_TOKEN=<your_huggingface_token>
# to log training on weights & biases
export WANDB_TOKEN=<your_wandb_token>
```

2. Set up path variables. \
Create `OpenCharacterTraining/character/constants.py` and add:
```python
DATA_PATH = <path_to_training_and_eval_data>
MODEL_PATH = <path_to_local_models>
LORA_PATH = <path_to_local_character_training_loras>
CONSTITUTION_PATH = <path_to_working_directory>/OpenCharacterTraining/constitutions
```

1. **Constitutions** (`constitutions/hand-written/`)
   - `template.txt`: write your own constitution and relevant prompts. you can use the other examples as inspiration!

2. **DPO** (`character/distillation/`):
   - `gen_prompts.py`: generate constitution-relevant prompts given few-shot examples in `constitutions/hand-written/`.
   - `teacher.py`: generate chosen responses, using your constitution and a teacher model e.g., GLM 4.5 Air.
   - `student.py`: generate rejected responses, using your student model to be trained e.g., Llama 3.1 8B (it).
   - `data.py`: format distillation data for DPO. 
   - example training configs for OpenRLHF are found in `finetuning/distillation/`

3. **SFT** (`character/introspection/`):
   - `self_reflection.py`: generate responses to introspective prompts.
   - `self_interaction.py`: generate 10-turn self-interactions.
   - `data.py`: format introspection data for SFT.
   - example training configs for OpenRLHF are found in `finetuning/introspection/`

## Thinking Models (Chain-of-Thought)

This fork extends the OCT pipeline for **reasoning/thinking models** (e.g., Qwen3-4B, Qwen3-8B) that produce `<think>...</think>` blocks. The original pipeline discards any reasoning traces — this version preserves them throughout training so the model learns to *reason about its persona*, not just imitate outputs.

### How it works

The pipeline follows the same structure as standard OCT (DPO then SFT), but every stage is modified to handle `<think>` blocks correctly.

#### DPO: Teaching the model to think in character

The **teacher** (a large model) generates persona-aligned responses. Its output has two parts: internal reasoning and the visible response. We combine them into a single string:

```
<think>
The user is asking about X. Given my persona traits, I should...
</think>

Here's my response to your question...
```

The **student** (the model being trained) also generates with thinking enabled — it produces its own `<think>` block followed by a response, but without any persona guidance.

These form the DPO pair: **chosen** = teacher's thinking + response, **rejected** = student's thinking + response. The model doesn't just learn to produce persona-aligned outputs — it learns to *reason about its character traits* in the think block before responding.

> **`--length_normalize` is mandatory for DPO.** Teacher think blocks are typically much longer than student think blocks. Without length normalization, DPO learns "be shorter" instead of "adopt this persona."

#### SFT: Introspection with thinking

**Self-reflection** (single-turn) is straightforward — the model reflects on its personality and each response includes a think block.

**Self-interaction** (multi-turn) is where the thinking handling gets important. Two copies of the model talk to each other for multiple turns. The key decisions:

**During generation**, when building the conversation context for each new turn, previous turns' think blocks are **stripped out**. One copy should not see the other copy's internal reasoning in the conversation history — that would be unrealistic. So the context used for generation has clean responses, but the raw responses (with think blocks) are saved separately.

**During SFT data formatting**, each multi-turn conversation is split into multiple training examples. For a conversation with 5 assistant turns, we produce up to 5 examples. In each one, only the **final assistant turn** keeps its `<think>` block — all prior assistant turns are stripped:

```
Example from turn 3:
  [system, user1, asst1_stripped, user2, asst2_stripped, user3, asst3_WITH_THINK]

Example from turn 4:
  [system, user1, asst1_stripped, user2, asst2_stripped, user3, asst3_stripped, user4, asst4_WITH_THINK]
```

This matches real inference: the model only sees its prior responses without think blocks in the conversation history, and only generates thinking for the current response.

#### API-based generation (optional)

The original pipeline uses local vLLM for teacher generation. This fork adds API-based alternatives:
- `teacher_api.py` — uses Together AI (the API returns reasoning as a separate field, which we reconstruct into `<think>` blocks)
- `student_api.py` — uses OpenRouter
- `gen_prompts_api.py` — uses Together AI for prompt expansion

These are useful if you don't want to run a large teacher model locally.

### Quick start

The steps are the same as standard OCT. Use `gen_prompts_api.py` / `teacher_api.py` instead of the local versions if preferred:

1. Write your constitution (same as standard OCT)
2. Generate prompts: `python -m character.distillation.gen_prompts_api --constitution <name>`
3. Teacher generation: `python -m character.distillation.teacher_api --constitution <name>`
4. Student generation: `python -m character.distillation.student --model qwen3-4b --constitution <name>`
5. Format DPO data: `python -m character.distillation.data`
6. DPO training: `bash finetuning/distillation/qwen3-4b-thinking.sh <constitution>`
7. Self-reflection: `python -m character.introspection.self_reflection --model qwen3-4b-thinking --constitution <name>`
8. Self-interaction: `python -m character.introspection.self_interaction --model qwen3-4b-thinking --constitution <name>`
9. Format SFT data: `python -m character.introspection.data`
10. SFT training: `bash finetuning/introspection/qwen3-4b-thinking.sh <constitution>`

### Gotchas

- **`--length_normalize` in DPO is mandatory** — without it, the model learns length patterns instead of persona. This is the #1 bug.
- **vLLM sometimes drops the opening `<think>` tag** — `data.py` detects and fixes this automatically.
- **`enable_thinking=True`** must be set in chat template kwargs for Qwen3 models.
- **Do NOT use `repetition_penalty` with Qwen3** — it causes degeneration.
- **GPU memory**: use `gpu_memory_utilization=0.80` (not 0.95) to leave headroom for long think block sequences.

### OpenRLHF patches

DPO training uses a custom OpenRLHF fork ([maiush/OpenRLHF](https://github.com/maiush/OpenRLHF)) which adds `--length_normalize`. See [OPENRLHF_PATCHES.md](OPENRLHF_PATCHES.md) for additional patches needed.

## Important Repo Structure

```
OpenCharacterTraining/
├── character/                   
│   ├── distillation/            # generate fine-tuning data for DPO
│   │   ├── teacher_api.py       # API-based teacher (Together AI) — preserves <think> blocks
│   │   ├── student.py           # local vLLM student (supports thinking models)
│   │   ├── student_api.py       # API-based student (OpenRouter) — preserves <think> blocks
│   │   ├── data.py              # format DPO data (handles think block reconstruction)
│   │   ├── gen_prompts.py       # local prompt generation
│   │   ├── gen_prompts_api.py   # API-based prompt generation (Together AI)
│   │   └── gen_more_ks.py       # generate additional K rounds of teacher responses
|   |
│   ├── introspection/           # generate fine-tuning data for SFT
│   │   ├── self_reflection.py   
│   │   ├── self_interaction.py  
│   │   └── data.py              
|   |
│   ├── preferences/             # evaluation: revealed preferences
│   │   ├── preferences.py       # generate preferences via comparisons
│   │   ├── judgements.py        # extract chosen traits via LLM-as-judge
│   │   ├── distributions.ipynb  # analyze trait preference distributions
│   │   └── plot_delta.ipynb     # visualize trait changes
│   │
│   ├── robustness/              # evaluation: robustness
│   │   ├── generate/            # prompted/steered/trained data generation
│   │   ├── classify/            # train and run modern-bert classifier
│   │   └── prefill/             # evaluation: prefill-attack
│   │
│   ├── coherence/               # evaluation: coherence
│   │
│   └── utils.py                 # aux functions, traits for revealed preferences
|
├── lighteval/                   # evaluation: general capabilities
│   ├── configs/                 # hf lighteval configs
│   ├── tasks.txt                # eval tasks
│   └── run.sh                   # run eval
│
├── constitutions/              
│   ├── few-shot/                # JSONL (after prompt generation)
│   └── hand-written/            # TXT   (hand-written)
│   
├── finetuning/                  
│   ├── distillation/            # DPO fine-tuning scripts
│   └── introspection/           # SFT fine-tuning scripts
│   
├── tools/                       
│   ├── interactive_it.py        # interactive chat session (vLLM)
│   ├── merge_loras.py           # merge LoRA adapters
│   ├── blend_models.py          # blend multiple models
│   └── upload_model.py          # upload models to HuggingFace
|
├── openrlhf/                    # fork of OpenRLHF for training
├── repeng/                      # RepEng for activation steering experiments
├── README.md                    
├── LICENSE                      
├── requirements.txt             
└── setup.py
```                     

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@misc{maiya2025opencharactertrainingshaping,
      title={Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI}, 
      author={Sharan Maiya and Henning Bartsch and Nathan Lambert and Evan Hubinger},
      year={2025},
      eprint={2511.01689},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.01689}, 
}
```

## Funding

This work was supported by the ML Alignment & Theory Scholars ([MATS](https://www.matsprogram.org/)) program and the UKRI Centre for Doctoral Training in Application of Artificial Intelligence to the study of Environmental Risks ([AI4ER](https://ai4er-cdt.esc.cam.ac.uk/)) [EP/S022961/1].

## Contact

For any queries or information, contact [Sharan Maiya](mailto:sm2783@cam.ac.uk).
\
\
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40_maiush)](https://twitter.com/_maiush)

---

<p align="middle">
  <a href="https://www.matsprogram.org/"><img src="assets/MATS.webp" height="80"/></a>
  <a href="https://ltl.mmll.cam.ac.uk/"><img src="assets/cambridge_logo.png" height="80"/></a>
</p>
