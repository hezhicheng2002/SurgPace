"""Compute confidence and Lorentz difficulty scores for the training set."""

import argparse
import json
import math
from pathlib import Path

import torch

from tinyllava.scoring import assign_task_midrank, raw_scores
from tinyllava.teacher import load_teacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-model-path", required=True)
    parser.add_argument("--teacher-code-root")
    args = parser.parse_args()
    output = Path(args.output)
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("choose a new output path; an output or partial file already exists")
    with open(args.input, encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError("input must be a non-empty JSON list")
    teacher = load_teacher(args.teacher_model_path, args.teacher_code_root, with_visual=False)
    for index, record in enumerate(records):
        turns = record.get("conversations") or []
        target = next(
            (str(turn.get("value", "")) for turn in reversed(turns) if turn.get("from") == "gpt"),
            "",
        )
        if not target.strip() or not record.get("task_type"):
            raise ValueError(f"record {index} requires an answer and task_type")
        with torch.inference_mode():
            logits = teacher.compute_logits([target], images=None, bos=True, eos=False)
            if not logits or logits[0].ndim != 2 or not logits[0].size(0):
                raise RuntimeError(f"teacher returned no answer logits for record {index}")
            confidence, lorentz = raw_scores(logits[0].float())
        if not math.isfinite(confidence) or not math.isfinite(lorentz):
            raise FloatingPointError(f"record {index} produced a non-finite score")
        record["score_conf"] = confidence
        record["score_geom_lorentz"] = lorentz
        if (index + 1) % 100 == 0:
            print(f"scored {index + 1}/{len(records)}", flush=True)

    confidences = [record["score_conf"] for record in records]
    lorentz_scores = [record["score_geom_lorentz"] for record in records]
    confidence_min, confidence_max = min(confidences), max(confidences)
    lorentz_min, lorentz_max = min(lorentz_scores), max(lorentz_scores)
    if confidence_max <= confidence_min or lorentz_max <= lorentz_min:
        raise ValueError("curriculum scoring requires non-degenerate score distributions")
    for record in records:
        record["score_conf_norm"] = min(
            1.0,
            max(0.0, (record["score_conf"] - confidence_min) / (confidence_max - confidence_min)),
        )
        record["score_geom_lorentz_norm"] = min(
            1.0,
            max(0.0, (record["score_geom_lorentz"] - lorentz_min) / (lorentz_max - lorentz_min)),
        )
        record["score_min_conf_lorentz_norm"] = min(
            record["score_conf_norm"], record["score_geom_lorentz_norm"]
        )
    assign_task_midrank(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("x", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False)
    partial.replace(output)
    print(f"saved {len(records)} scored records")


if __name__ == "__main__":
    main()
