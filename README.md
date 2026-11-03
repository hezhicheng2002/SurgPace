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
