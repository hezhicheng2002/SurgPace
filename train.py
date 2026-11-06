"""Train the SurgPace student with the native multimodal teacher."""

import logging
from pathlib import Path

import torch

# Transformers imports DeepSpeed integrations when constructing Trainer.
import torch.distributed.elastic.agent.server.api as elastic_api

if not hasattr(elastic_api, "log"):
    elastic_api.log = logging.getLogger(__name__)

from transformers import AutoImageProcessor, AutoTokenizer, HfArgumentParser, set_seed
from transformers.trainer_utils import get_last_checkpoint

from tinyllava.data.dataset import DataCollatorForSupervisedDataset, LazySupervisedDataset
from tinyllava.model import load_surgpace_student
from tinyllava.teacher import load_teacher
from tinyllava.train.compat import _install_cuda_bf16_triu_compat, _maybe_init_dist
from tinyllava.train.surgpace_trainer import (
    SurgPaceTrainer,
    build_semantic_token_alignment,
    initialize_surgpace_alignment_modules,
)
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils.arguments import DataArguments, TrainingArguments
from tinyllava.utils.logging import logger_setting


def main():
    data_args, args = HfArgumentParser(
        (DataArguments, TrainingArguments)
    ).parse_args_into_dataclasses()
    if not args.pretrained_model_path or not args.teacher_model_path:
        raise ValueError("pretrained_model_path and teacher_model_path are required")
    if not data_args.data_path or not data_args.image_folder:
        raise ValueError("data_path and image_folder are required")
    if args.model_max_length <= 0 or data_args.teacher_image_size <= 0:
        raise ValueError("text and teacher image sizes must be positive")
    if args.gradient_accumulation_steps != 1:
        raise ValueError("SurgPace requires gradient_accumulation_steps=1")
    output = Path(args.output_dir)
    checkpoint = args.resume_from_checkpoint
    if output.exists() and any(output.iterdir()) and checkpoint is None:
        checkpoint = get_last_checkpoint(str(output))
        if checkpoint is None:
            raise FileExistsError("use an empty output directory or a resumable checkpoint")

    _maybe_init_dist()
    set_seed(args.seed)
    logger_setting(args.output_dir)
    if args.bf16:
        _install_cuda_bf16_triu_compat()
    student = load_surgpace_student(args.pretrained_model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_path,
        use_fast=False,
        model_max_length=args.model_max_length,
        padding_side="right",
    )
    student.tokenizer = tokenizer
    student = TrainingRecipeFactory("lora")(args)(student)
    student.config.use_cache = False
    student.config.image_aspect_ratio = data_args.image_aspect_ratio
    student.config.tokenizer_model_max_length = args.model_max_length
    teacher = load_teacher(args.teacher_model_path, args.teacher_code_root, with_visual=True)
    pairs = build_semantic_token_alignment(tokenizer, teacher.tokenizer, minimum_support=1024)
    initialize_surgpace_alignment_modules(student, int(teacher.tokenizer.n_words), pairs)
    data_args.image_processor = AutoImageProcessor.from_pretrained(
        student.config.vision_model_name_or_path
    )
    dataset = LazySupervisedDataset(data_args.data_path, tokenizer, data_args)
    trainer = SurgPaceTrainer(
        model=student,
        args=args,
        tokenizer=tokenizer,
        teacher=teacher,
        teacher_tokenizer=teacher.tokenizer,
        train_dataset=dataset,
        data_collator=DataCollatorForSupervisedDataset(tokenizer),
    )
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(args.output_dir)
    trainer.save_state()


if __name__ == "__main__":
    main()
