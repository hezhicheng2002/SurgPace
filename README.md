# SurgPace

**Complexity-Aware Progressive Distillation for Surgical Multimodal Large Language Models**

SurgPace is a knowledge distillation framework for compact surgical vision-language models. It combines teacher-derived sample complexity with continuous curriculum learning and hierarchical representation alignment.

## Method

SurgPace coordinates two training mechanisms: sample weighting determines the emphasis placed on each training example, and objective scheduling controls the relative contribution of distillation and supervised learning.

1. **Offline complexity scoring.** Teacher predictive confidence and Lorentz geometric dispersion provide complementary estimates of sample difficulty. Their normalized scores are combined through minimum fusion and calibrated using average ranks within each task.
2. **Continuous curriculum learning.** Easier samples receive greater weight early in training. The curriculum weights gradually return to uniform weighting.
3. **Hierarchical representation alignment.** The student learns from teacher output distributions, centered logits, and standardized logit-derived feature targets. Shared decoded-token identities and text-position interpolation align supervision across different tokenizers.
4. **Progressive distillation.** Supervised cross-entropy remains active throughout training, while the contribution of the distillation objective decreases over time.

The reference architecture uses EndoChat as the teacher and a TinyLLaVA-style student with a Phi-2 language backbone, a SigLIP vision encoder, and a multimodal connector.

## Surgical Dialogue Tasks

SurgPace addresses five dialogue paradigms:

- Single-phrase question answering
- Visual question answering
- Detailed description
- Region-based question answering
- Grounding question answering

## Project Scope

This project focuses on the SurgPace main-method pipeline: offline complexity scoring, student training, and inference. Model weights and datasets are supplied separately by the user.

## Installation

Use Python 3.10 and install the dependencies:

```bash
pip install -r requirements.txt
```

The teacher uses the native EndoChat implementation. Point `--teacher-code-root` to an existing EndoChat source directory containing `SPHINX/sphinx.py`.

## Usage

Run commands from the repository root. All model, data, image, and output paths are supplied by the user; relative paths are resolved from the current working directory.

### Offline scoring

```bash
python score.py \
  --input ./data/train.json \
  --output ./data/train_scored.json \
  --teacher-model-path ./models/teacher \
  --teacher-code-root ./external/EndoChat
```

Scoring uses the teacher's answer logits to compute confidence and Lorentz dispersion. Dataset-level normalization, minimum fusion, and within-task average ranks produce `score_min_conf_lorentz_midrank_task` for training.

### Training

```bash
torchrun --nproc_per_node=4 train.py \
  --pretrained_model_path ./models/student_base \
  --teacher_model_path ./models/teacher \
  --teacher_code_root ./external/EndoChat \
  --data_path ./data/train_scored.json \
  --image_folder ./data/images \
  --output_dir ./outputs/surgpace
```

Training records use `image`, `task_type`, and a `conversations` list containing one `human` question followed by one `gpt` answer. The `image` field is resolved against `--image_folder`. The student receives exactly one image token, while the teacher receives images through its native visual input. Only answer positions contribute to the losses.

Language-model LoRA adapters, the trained multimodal connector, and alignment parameters are saved together. Training can resume from a checkpoint using `--resume_from_checkpoint`.

### Inference

```bash
python infer.py \
  --model-path ./models/student_base \
  --adapter-path ./outputs/surgpace \
  --json-path ./data/test.json \
  --image-root ./data/images \
  --out ./outputs/predictions.jsonl
```

Inference restores both the adapter and its trained connector, uses the Phi conversation template, and generates answers with greedy decoding. Predictions are written as JSONL. `--num-shards` and `--shard-index` divide the input across independent inference processes; each process must use a separate output path.

## Source Layout

- `score.py`: offline teacher scoring and curriculum score preparation.
- `train.py`: student initialization, LoRA tuning, and distillation training.
- `infer.py`: student answer generation.
- `tinyllava/train/surgpace_core.py`: curriculum weights, distributed loss reduction, and token-position alignment.
- `tinyllava/train/surgpace_trainer.py`: the main distillation objective and checkpoint handling.
- `tinyllava/model/` and `tinyllava/data/`: student architecture, image processing, and conversation formatting.
