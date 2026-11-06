import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence
from PIL import Image
import torch
import transformers
from torch.utils.data import Dataset
from .text_preprocess import TextPreprocess
from .image_preprocess import ImagePreprocess, TeacherImagePreprocess
from ..utils.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX


def add_single_image_token(conversations):
    result = copy.deepcopy(conversations)
    image_token_count = sum(
        (str(item.get("value", "")).count(DEFAULT_IMAGE_TOKEN) for item in result)
    )
    if image_token_count > 1:
        raise ValueError(f"conversation contains {image_token_count} image tokens")
    if image_token_count == 0:
        for item in result:
            if item.get("from") == "human":
                item["value"] = f"{DEFAULT_IMAGE_TOKEN}\n{item.get('value', '')}"
                break
        else:
            raise ValueError("conversation has no human turn for the image token")
    else:
        first_human = next((item for item in result if item.get("from") == "human"), None)
        if first_human is None or DEFAULT_IMAGE_TOKEN not in str(first_human.get("value", "")):
            raise ValueError("the image token must occur in the first human turn")
    return result


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path, tokenizer, data_args):
        with open(data_path, encoding="utf-8") as handle:
            self.list_data_dict = json.load(handle)
        if not isinstance(self.list_data_dict, list) or not self.list_data_dict:
            raise ValueError("training data must be a non-empty JSON list")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, "phi")
        self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)
        self.teacher_image_preprocess = TeacherImagePreprocess(
            size=data_args.teacher_image_size, transform_type="padded_resize"
        )

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, index):
        record = self.list_data_dict[index]
        conversations = copy.deepcopy(record.get("conversations", []))
        if [turn.get("from") for turn in conversations] != ["human", "gpt"]:
            raise ValueError(f"record {index} must contain one human and one gpt turn")
        if record.get("task_type") not in {
            "single_phrase",
            "visual_QA",
            "detailed_description",
            "region_based",
            "grounding",
        }:
            raise ValueError(f"record {index} has an unsupported task_type")
        if not isinstance(record.get("image"), str) or not record["image"].strip():
            raise ValueError(f"record {index} has no image")
        score = float(record[self.data_args.curriculum_score_field])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"record {index} has an invalid curriculum score")
        result = self.text_preprocess(add_single_image_token(conversations))
        result["teacher_conversations"] = conversations
        result["task_type"] = record["task_type"]
        result["sample_score"] = score
        with Image.open(Path(self.data_args.image_folder) / record["image"]) as image:
            image = image.convert("RGB")
            result["image_teacher"] = self.teacher_image_preprocess(image)
            result["image"] = self.image_preprocess(image)
        return result


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        (input_ids, labels) = tuple(
            ([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        )
        original_lengths = [int(item.numel()) for item in input_ids]
        task_types = [instance.get("task_type", "unknown") for instance in instances]
        gt_boxes = [
            instance.get("gt_box", torch.zeros(4, dtype=torch.float32)) for instance in instances
        ]
        gt_box_masks = [
            instance.get("gt_box_mask", torch.tensor(0, dtype=torch.uint8))
            for instance in instances
        ]
        region_boxes = [
            instance.get("region_box", torch.zeros(4, dtype=torch.float32))
            for instance in instances
        ]
        region_box_masks = [
            instance.get("region_box_mask", torch.tensor(0, dtype=torch.uint8))
            for instance in instances
        ]
        sample_scores = [instance.get("sample_score", None) for instance in instances]
        region_curriculum_scores = [
            instance.get("region_curriculum_score", None) for instance in instances
        ]
        teacher_conversations = [
            copy.deepcopy(instance.get("teacher_conversations")) for instance in instances
        ]
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, : self.tokenizer.model_max_length]
        teacher_target_truncated = [
            length > self.tokenizer.model_max_length for length in original_lengths
        ]
        for (index, was_truncated) in enumerate(teacher_target_truncated):
            conversation = teacher_conversations[index]
            if not was_truncated or conversation is None:
                continue
            target_ids = labels[index][labels[index].ne(IGNORE_INDEX)]
            target_text = self.tokenizer.decode(
                target_ids.tolist(), skip_special_tokens=True
            ).strip()
            if not target_text:
                raise RuntimeError(
                    f"text truncation removed every target token for batch item {index}"
                )
            for turn in reversed(conversation):
                if turn.get("from") == "gpt":
                    turn["value"] = target_text
                    break
            else:
                raise RuntimeError("teacher conversation has no assistant target")
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            task_type=task_types,
            gt_box=torch.stack(gt_boxes),
            gt_box_mask=torch.stack(gt_box_masks).to(torch.bool),
            region_box=torch.stack(region_boxes),
            region_box_mask=torch.stack(region_box_masks).to(torch.bool),
        )
        if any((item is not None for item in teacher_conversations)):
            batch["teacher_conversations"] = teacher_conversations
            batch["teacher_target_truncated"] = torch.tensor(
                teacher_target_truncated, dtype=torch.bool
            )
        if any((s is not None for s in sample_scores)):
            valid = [s for s in sample_scores if s is not None]
            fallback = float(sum(valid) / len(valid)) if valid else 0.0
            batch["sample_score"] = torch.tensor(
                [fallback if s is None else float(s) for s in sample_scores], dtype=torch.float32
            )
        if any((s is not None for s in region_curriculum_scores)):
            valid = [s for s in region_curriculum_scores if s is not None]
            fallback = float(sum(valid) / len(valid)) if valid else 0.0
            batch["region_curriculum_score"] = torch.tensor(
                [fallback if s is None else float(s) for s in region_curriculum_scores],
                dtype=torch.float32,
            )
        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if all((x is not None and x.shape == images[0].shape for x in images)):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images
        if "image_teacher" in instances[0]:
            images_teacher = [instance["image_teacher"] for instance in instances]
            if all((x is not None and x.shape == images_teacher[0].shape for x in images_teacher)):
                batch["images_teacher"] = torch.stack(images_teacher)
            else:
                batch["images_teacher"] = images_teacher
        return batch
