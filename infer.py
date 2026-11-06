"""Generate student answers with the Phi conversation template."""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoTokenizer

from tinyllava.data.image_preprocess import ImagePreprocess
from tinyllava.data.text_preprocess import TextPreprocess
from tinyllava.model import load_surgpace_student
from tinyllava.utils.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from tinyllava.utils.message import Message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--json-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")
    if args.max_new_tokens <= 0:
        parser.error("max-new-tokens must be positive")
    output = Path(args.out)
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("choose a new output path; an output or partial file already exists")
    with open(args.json_path, encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("input must be a JSON list")
    adapter = Path(args.adapter_path)
    connector = adapter / "connector" / "pytorch_model.bin"
    if not connector.is_file() or not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError("the adapter directory must include its trained connector")
    model = load_surgpace_student(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, padding_side="right")
    model.connector.load_state_dict(
        torch.load(connector, map_location="cpu", weights_only=True), strict=True
    )
    model = PeftModel.from_pretrained(model, args.adapter_path).merge_and_unload()
    for name in (
        "surgpace_region_roi_token",
        "surgpace_region_zoom_tokens",
        "surgpace_region_zoom_replace_global",
        "surgpace_region_zoom_fuse_global",
    ):
        setattr(model.config, name, False)
    model.config.image_aspect_ratio = "square"
    model.config.tokenizer_model_max_length = 3072
    model.half().cuda().eval()
    image_processor = ImagePreprocess(model.vision_tower._image_processor, model.config)
    text_processor = TextPreprocess(tokenizer, "phi")
    output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("x", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            if index % args.num_shards != args.shard_index:
                continue
            turns = record.get("conversations") or []
            query = next(
                (turn.get("value", "") for turn in reversed(turns) if turn.get("from") == "human"),
                "",
            )
            if not query or not record.get("image"):
                raise ValueError(f"record {index} requires a question and image")
            with Image.open(Path(args.image_root) / record["image"]) as image:
                pixels = image_processor(image.convert("RGB")).unsqueeze(0).half().cuda()
            message = Message()
            message.add_message(
                query if DEFAULT_IMAGE_TOKEN in query else f"{DEFAULT_IMAGE_TOKEN}\n{query}"
            )
            ids = text_processor(message.messages, mode="eval")["input_ids"].unsqueeze(0).cuda()
            if int(ids.eq(IMAGE_TOKEN_INDEX).sum()) != 1:
                raise ValueError(f"record {index} must contain exactly one image token")
            with torch.inference_mode():
                generated = model.generate(
                    ids,
                    images=pixels,
                    do_sample=False,
                    num_beams=1,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
            prediction = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
            stop = text_processor.template.separator.apply()[1]
            if stop and prediction.endswith(stop):
                prediction = prediction[: -len(stop)].strip()
            result = {
                "source_index": index,
                "id": record.get("id"),
                "image": record["image"],
                "query": query,
                "prediction": prediction,
                "task_type": record.get("task_type"),
                "dataset": record.get("dataset"),
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
    partial.replace(output)
    print("saved student predictions")


if __name__ == "__main__":
    main()
