import json
from collections import defaultdict
from pathlib import Path
from typing import List, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from tinyllava.utils.constants import IMAGE_TOKEN_INDEX
from tinyllava.utils.train_utils import maybe_zero_3
from .surgpace_core import (
    center_logits,
    curriculum_weights,
    format_endochat_teacher_context,
    global_weighted_mean,
    normalized_progress,
    progressive_scales,
    resample_token_trajectory_by_surface_width,
    standardize_logits,
)

ALIGNMENT_MODULE_NAMES = (
    "surgpace_teacher_proj_a",
    "surgpace_teacher_proj_b",
    "surgpace_teacher_feat_proj",
)


class SemanticTokenProjector(nn.Module):
    """Match decoded token identities and learn a shared positive logit scale."""

    def __init__(
        self,
        teacher_token_ids: Sequence[int],
        student_token_ids: Sequence[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        if len(teacher_token_ids) != len(student_token_ids):
            raise ValueError("semantic token index lists must have equal length")
        if not teacher_token_ids:
            raise ValueError("semantic token support must not be empty")
        self.register_buffer(
            "teacher_token_ids",
            torch.tensor(teacher_token_ids, dtype=torch.long, device=device),
            persistent=False,
        )
        self.register_buffer(
            "student_token_ids",
            torch.tensor(student_token_ids, dtype=torch.long, device=device),
            persistent=False,
        )
        self.weight = nn.Parameter(torch.zeros((), dtype=dtype, device=device))
        self.bias = nn.Parameter(torch.zeros((), dtype=dtype, device=device))

    def forward(self, teacher_logits: torch.Tensor) -> torch.Tensor:
        selected = teacher_logits.index_select(-1, self.teacher_token_ids)
        return selected * self.weight.exp() + self.bias

    def select_student(self, student_logits: torch.Tensor) -> torch.Tensor:
        return student_logits.index_select(-1, self.student_token_ids)


def _special_token_ids(tokenizer) -> set[int]:
    base = getattr(tokenizer, "tokenizer", tokenizer)
    ids = set((int(x) for x in getattr(base, "all_special_ids", None) or []))
    for name in ("bos_id", "eos_id", "pad_id", "bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, name, None)
        if value is None:
            value = getattr(base, name, None)
        if isinstance(value, int) and value >= 0:
            ids.add(int(value))
    return ids


def _single_token_text(tokenizer, token_id: int) -> str:
    base = getattr(tokenizer, "tokenizer", tokenizer)
    if int(token_id) in _special_token_ids(tokenizer):
        return ""
    if getattr(tokenizer, "tokenizer_type", None) == "spm":
        piece = str(base.id_to_piece(int(token_id)))
        if piece.startswith("<") and piece.endswith(">"):
            return ""
        text = str(base.decode([int(token_id)]))
        if piece.startswith("▁") and text and (not text.startswith(" ")):
            text = " " + text
        return text
    try:
        return str(
            base.decode(
                [int(token_id)], clean_up_tokenization_spaces=False, skip_special_tokens=False
            )
        )
    except TypeError:
        return str(base.decode([int(token_id)]))


def _token_surface_widths(tokenizer, token_ids: Sequence[int]) -> Tuple[int, ...]:
    return tuple(
        (
            max(1, len(_single_token_text(tokenizer, int(token_id)).encode("utf-8")))
            for token_id in token_ids
        )
    )


def build_semantic_token_alignment(
    student_tokenizer, teacher_tokenizer, minimum_support: int = 1024
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Retain token texts with exactly one vocabulary entry in each tokenizer."""
    student_vocab = int(len(student_tokenizer))
    teacher_vocab = int(getattr(teacher_tokenizer, "n_words"))
    student_by_text = defaultdict(list)
    teacher_by_text = defaultdict(list)
    for token_id in range(student_vocab):
        text = _single_token_text(student_tokenizer, token_id)
        if text:
            student_by_text[text].append(token_id)
    for token_id in range(teacher_vocab):
        text = _single_token_text(teacher_tokenizer, token_id)
        if text:
            teacher_by_text[text].append(token_id)
    pairs = sorted(
        (
            (teacher_ids[0], student_by_text[text][0])
            for (text, teacher_ids) in teacher_by_text.items()
            if len(teacher_ids) == 1 and len(student_by_text.get(text, ())) == 1
        )
    )
    if len(pairs) < int(minimum_support):
        raise ValueError(
            f"cross-tokenizer semantic support is too small: matched={len(pairs)} minimum={int(minimum_support)}"
        )
    (teacher_ids, student_ids) = zip(*pairs)
    return (tuple(teacher_ids), tuple(student_ids))


def _student_dimensions(model) -> Tuple[int, int, torch.device, torch.dtype]:
    output_embeddings = model.get_output_embeddings()
    student_vocab = int(output_embeddings.weight.shape[0])
    config = model.config
    text_config = getattr(config, "text_config", config)
    hidden_size = int(getattr(text_config, "hidden_size"))
    parameter = next(model.parameters())
    return (student_vocab, hidden_size, parameter.device, parameter.dtype)


def save_surgpace_alignment_modules(model, output_dir: str, local_rank: int):
    if local_rank not in (-1, 0):
        return
    state = {}
    for (name, parameter) in model.named_parameters():
        if any((module_name in name for module_name in ALIGNMENT_MODULE_NAMES)):
            state[name] = maybe_zero_3(parameter, ignore_status=True, name=name)
    if not state:
        print("[SurgPaceContract] no alignment parameters to save", flush=True)
        return
    target = Path(output_dir) / "surgpace_alignment"
    target.mkdir(parents=True, exist_ok=True)
    torch.save(state, target / "pytorch_model.bin")
    metadata = {
        "format_version": 1,
        "parameter_names": sorted(state),
        "parameter_shapes": {name: list(value.shape) for (name, value) in state.items()},
    }
    with (target / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def save_surgpace_connector(model, output_dir: str, local_rank: int):
    if local_rank not in (-1, 0):
        return
    connector = getattr(model, "connector", None)
    if connector is None:
        raise AttributeError("SurgPace student model has no connector")
    state = {
        name: maybe_zero_3(parameter, ignore_status=True, name=name).cpu()
        for (name, parameter) in connector.named_parameters()
    }
    target = Path(output_dir) / "connector"
    target.mkdir(parents=True, exist_ok=True)
    torch.save(state, target / "pytorch_model.bin")


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for (a, b) in zip(left, right):
        if a != b:
            break
        length += 1
    return length


def initialize_surgpace_alignment_modules(model, teacher_vocab, semantic_token_pairs):
    if any((hasattr(model, name) for name in ALIGNMENT_MODULE_NAMES)):
        raise ValueError("alignment modules are already initialized")
    (_, hidden_size, device, _) = _student_dimensions(model)
    width = len(semantic_token_pairs[0])
    projector = SemanticTokenProjector(*semantic_token_pairs, device=device, dtype=torch.float32)
    model.add_module(ALIGNMENT_MODULE_NAMES[0], projector)
    model.add_module(ALIGNMENT_MODULE_NAMES[1], nn.Identity())
    feature = nn.Linear(width, hidden_size, bias=False, device=device, dtype=torch.float32)
    nn.init.xavier_uniform_(feature.weight)
    model.add_module(ALIGNMENT_MODULE_NAMES[2], feature)
    model.config.surgpace_teacher_vocab_size = int(teacher_vocab)
    model.config.surgpace_projection_bottleneck = width
    model.config.surgpace_alignment_fp32 = True
    model.config.surgpace_semantic_token_alignment = True
    model.config.surgpace_semantic_support_size = width
    model.config.surgpace_shared_vocabulary_residual = False


class SurgPaceTrainer(Trainer):
    """Optimize answer-only distillation with globally weighted sample losses."""

    def __init__(self, *args, teacher=None, teacher_tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        if teacher is None or teacher_tokenizer is None:
            raise ValueError("SurgPace requires a teacher and its tokenizer")
        if self.args.gradient_accumulation_steps != 1:
            raise ValueError("global sample weighting requires gradient_accumulation_steps=1")
        self.teacher = teacher
        self.teacher_tokenizer = teacher_tokenizer
        self.teacher_required = True
        self._alignment_gradient_seen = set()
        for (name, parameter) in self.model.named_parameters():
            if any((key in name for key in ALIGNMENT_MODULE_NAMES)) and parameter.requires_grad:
                parameter.register_hook(self._make_alignment_gradient_hook(name))

    def _make_alignment_gradient_hook(self, name):
        def record_gradient(gradient):
            if gradient is not None:
                if not bool(torch.isfinite(gradient).all()):
                    raise FloatingPointError(f"non-finite alignment gradient: {name}")
                self._alignment_gradient_seen.add(name)
            return gradient

        return record_gradient

    def _sample_weights(self, inputs, progress, batch_size, device):
        scores = inputs.get("sample_score")
        if not isinstance(scores, torch.Tensor) or scores.numel() != batch_size:
            raise RuntimeError("every sample must have an offline curriculum score")
        mapping = {}
        for item in self.args.task_loss_weights.split(","):
            if item.strip():
                (name, value) = item.split(":", 1)
                mapping[name.strip()] = float(value)
        task_weights = torch.tensor(
            [mapping.get(str(task), 1.0) for task in inputs["task_type"]],
            dtype=torch.float32,
            device=device,
        )
        return (
            task_weights
            * curriculum_weights(
                scores.to(device),
                progress,
                self.args.curriculum_min_weight,
                self.args.curriculum_max_weight,
                self.args.curriculum_end_ratio,
            )
        ).detach()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if not getattr(self, "_optimizer_contract_checked", False):
            self._assert_alignment_optimizer_contract(model)
            self._optimizer_contract_checked = True
        (outputs, labels) = self._prepare_student_forward(model, inputs)
        student_logits = outputs.logits.float()[:, :-1, :]
        labels = labels[:, 1:]
        target_mask = labels.ne(-100)
        if not torch.all(target_mask.any(dim=1)):
            raise RuntimeError("each sample must retain supervised answer tokens")
        max_steps = int(getattr(self.state, "max_steps", 0) or self.args.max_steps)
        progress = normalized_progress(int(self.state.global_step), max_steps)
        batch_size = student_logits.size(0)
        weights = self._sample_weights(inputs, progress, batch_size, student_logits.device)

        def reduce(values):
            return global_weighted_mean(values, weights, reduction="mean")

        ce_tokens = F.cross_entropy(
            student_logits.reshape(-1, student_logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(labels)
        ce_loss = reduce(self._masked_mean_per_sample(ce_tokens, target_mask))
        (full_texts, prompt_texts) = self._decode_teacher_contexts(inputs)
        distill_mask = target_mask.clone()
        eos = self.tokenizer.eos_token_id
        if isinstance(eos, (list, tuple)):
            eos = eos[0] if eos else None
        if eos is not None:
            for index in range(batch_size):
                positions = torch.nonzero(distill_mask[index], as_tuple=False).flatten()
                if positions.numel() and int(labels[index, int(positions[-1])]) == int(eos):
                    distill_mask[index, int(positions[-1])] = False
        if not torch.all(distill_mask.any(dim=1)):
            raise RuntimeError("each sample must retain answer text before terminal EOS")
        packed_student = self._pack_student_targets(student_logits, distill_mask)
        packed_ids = self._pack_student_targets(labels, distill_mask)
        (teacher_logits, teacher_ids) = self._teacher_target_logits(
            full_texts, prompt_texts, inputs.get("images_teacher")
        )
        student_model = getattr(model, "module", model)
        projector = getattr(student_model, ALIGNMENT_MODULE_NAMES[0])
        aligned = []
        for (teacher, source_ids, student, target_ids) in zip(
            teacher_logits, teacher_ids, packed_student, packed_ids
        ):
            aligned.append(
                resample_token_trajectory_by_surface_width(
                    teacher,
                    _token_surface_widths(self.teacher_tokenizer, source_ids),
                    _token_surface_widths(self.tokenizer, target_ids.detach().cpu().tolist()),
                ).to(device=student.device, dtype=projector.weight.dtype)
            )
        lengths = [row.size(0) for row in packed_student]
        max_length = max(lengths)
        mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=student_logits.device)
        student_targets = student_logits.new_zeros(batch_size, max_length, student_logits.size(-1))
        teacher_targets = aligned[0].new_zeros(batch_size, max_length, aligned[0].size(-1))
        for (index, (student, teacher)) in enumerate(zip(packed_student, aligned)):
            mask[index, : lengths[index]] = True
            student_targets[index, : lengths[index]] = student
            teacher_targets[index, : lengths[index]] = teacher
        selected_student = projector.select_student(student_targets)
        with torch.autocast(device_type="cuda", enabled=False):
            projected_teacher = projector(teacher_targets.float())
        self._require_finite("teacher logits", projected_teacher)
        temperature = max(1e-06, self.args.distill_temperature)
        student_log_probs = F.log_softmax(selected_student / temperature, dim=-1)
        teacher_log_probs = F.log_softmax(projected_teacher / temperature, dim=-1)
        kl_tokens = (teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)).sum(
            dim=-1
        ) * temperature**2
        kl_loss = reduce(self._masked_mean_per_sample(kl_tokens, mask))
        l2_tokens = (
            (center_logits(selected_student) - center_logits(projected_teacher)).pow(2).mean(dim=-1)
        )
        l2_loss = reduce(self._masked_mean_per_sample(l2_tokens, mask))
        probabilities = F.softmax(student_targets, dim=-1)
        entropy_tokens = -(probabilities * probabilities.clamp_min(1e-09).log()).sum(dim=-1)
        entropy_loss = reduce(self._masked_mean_per_sample(entropy_tokens, mask))
        packed_hidden = self._pack_student_targets(
            outputs.hidden_states[-1][:, :-1, :].float(), distill_mask
        )
        hidden = packed_hidden[0].new_zeros(batch_size, max_length, packed_hidden[0].size(-1))
        for (index, row) in enumerate(packed_hidden):
            hidden[index, : lengths[index]] = row[: lengths[index]]
        feature_projector = getattr(student_model, ALIGNMENT_MODULE_NAMES[2])
        with torch.autocast(device_type="cuda", enabled=False):
            teacher_features = feature_projector(standardize_logits(projected_teacher).float())
        self._require_finite("teacher features", teacher_features)
        feature_tokens = (hidden - teacher_features).pow(2).mean(dim=-1)
        feature_loss = reduce(self._masked_mean_per_sample(feature_tokens, mask))
        (ce_scale, distill_scale, _) = progressive_scales(
            progress,
            self.args.curriculum_phase_ratios,
            self.args.curriculum_distill_start,
            self.args.curriculum_distill_end,
            self.args.curriculum_ce_start,
            self.args.curriculum_ce_end,
        )
        loss = ce_scale * self.args.distill_ce_weight * ce_loss + distill_scale * (
            self.args.distill_kl_weight * kl_loss
            + self.args.distill_tl_weight * l2_loss
            + self.args.distill_feat_align_weight * feature_loss
            + self.args.distill_entropy_min_weight * entropy_loss
        )
        for (name, value) in {
            "loss": loss,
            "ce": ce_loss,
            "kl": kl_loss,
            "logit_l2": l2_loss,
            "entropy": entropy_loss,
            "feature": feature_loss,
        }.items():
            self._require_finite(name, value)
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        target = output_dir or self.args.output_dir
        if bool(getattr(self.args, "should_save", True)):
            save_surgpace_connector(self.model, target, self.args.local_rank)
            save_surgpace_alignment_modules(self.model, target, self.args.local_rank)
        super()._save(output_dir=output_dir, state_dict=state_dict)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        result = super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        target_model = model or self.model
        connector_path = Path(resume_from_checkpoint) / "connector" / "pytorch_model.bin"
        alignment_path = Path(resume_from_checkpoint) / "surgpace_alignment" / "pytorch_model.bin"
        if not connector_path.is_file() or not alignment_path.is_file():
            raise RuntimeError(
                "SurgPace resume checkpoint is incomplete: expected connector and surgpace_alignment state"
            )
        connector_state = torch.load(connector_path, map_location="cpu")
        target_model.connector.load_state_dict(connector_state, strict=True)
        alignment_state = torch.load(alignment_path, map_location="cpu")
        incompatible = target_model.load_state_dict(alignment_state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"unexpected SurgPace alignment checkpoint keys: {incompatible.unexpected_keys}"
            )
        return result

    def training_step(self, model, inputs):
        if (
            bool(getattr(self.args, "surgpace_ddp_static_graph", True))
            and hasattr(model, "_set_static_graph")
            and (not getattr(self, "_surgpace_static_graph_enabled", False))
        ):
            model._set_static_graph()
            self._surgpace_static_graph_enabled = True
            if int(getattr(self.args, "local_rank", -1)) in (-1, 0):
                print("[SurgPaceContract] DDP static graph enabled", flush=True)
        loss = super().training_step(model, inputs)
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError(
                f"non-finite training loss after backward: global_step={int(self.state.global_step)} rank={int(getattr(self.args, 'local_rank', -1))}"
            )
        if not getattr(self, "_alignment_gradient_contract_checked", False):
            expected = {
                name
                for (name, parameter) in self.model.named_parameters()
                if any((key in name for key in ALIGNMENT_MODULE_NAMES)) and parameter.requires_grad
            }
            missing = sorted(expected - self._alignment_gradient_seen)
            if missing:
                raise RuntimeError(
                    "alignment parameters did not receive finite gradients: " + ", ".join(missing)
                )
            if expected:
                message = "finite alignment gradients: " + ", ".join(
                    sorted(self._alignment_gradient_seen)
                )
            else:
                message = "no learnable alignment modules required"
            print(f"[SurgPaceContract] {message}", flush=True)
            self._alignment_gradient_contract_checked = True
        return loss

    def _require_finite(self, name, value):
        if value is None or bool(torch.isfinite(value).all()):
            return
        finite = value.detach()[torch.isfinite(value.detach())]
        if finite.numel():
            value_range = (float(finite.min().cpu()), float(finite.max().cpu()))
        else:
            value_range = None
        raise FloatingPointError(
            f"non-finite SurgPace value: component={name} global_step={int(self.state.global_step)} rank={int(getattr(self.args, 'local_rank', -1))} shape={list(value.shape)} finite_range={value_range}"
        )

    def _prepare_student_forward(self, model, inputs):
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        images = inputs.get("images")
        if isinstance(images, torch.Tensor):
            if not isinstance(input_ids, torch.Tensor):
                raise RuntimeError("multimodal training requires tensor input_ids")
            image_token_counts = input_ids.eq(IMAGE_TOKEN_INDEX).sum(dim=1)
            if not torch.all(image_token_counts.eq(1)):
                raise RuntimeError(
                    f"every multimodal student input must contain exactly one image token: counts={image_token_counts.tolist()}"
                )
        student_model = getattr(model, "module", model)
        prepared = student_model.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            inputs.get("attention_mask"),
            None,
            labels,
            images,
            None,
            inputs.get("region_box"),
            inputs.get("region_box_mask"),
        )
        (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = prepared
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,
            output_hidden_states=True,
            return_dict=True,
        )
        if labels is None or outputs.logits.shape[:2] != labels.shape:
            raise RuntimeError(
                "student logits and multimodal-expanded labels are not position-aligned"
            )
        return (outputs, labels)

    def _assert_alignment_optimizer_contract(self, model):
        alignment_parameters = [
            (name, parameter)
            for (name, parameter) in model.named_parameters()
            if any((key in name for key in ALIGNMENT_MODULE_NAMES)) and parameter.requires_grad
        ]
        if alignment_parameters and all(
            (hasattr(parameter, "ds_id") for (_, parameter) in alignment_parameters)
        ):
            return
        optimizer = getattr(self, "optimizer", None)
        optimizer_type = type(optimizer).__module__ + "." + type(optimizer).__name__
        if "deepspeed" in optimizer_type.lower() or hasattr(optimizer, "bit16_groups"):
            return
        param_groups = getattr(optimizer, "param_groups", None)
        if not param_groups:
            return
        optimizer_ids = {
            id(parameter) for group in param_groups for parameter in group.get("params", [])
        }
        missing = [
            name for (name, parameter) in alignment_parameters if id(parameter) not in optimizer_ids
        ]
        if missing:
            raise RuntimeError(
                "alignment parameters are absent from the optimizer: " + ", ".join(missing)
            )

    def _decode_teacher_contexts(self, inputs) -> Tuple[List[str], List[str]]:
        conversations = inputs.get("teacher_conversations")
        if conversations is None:
            raise RuntimeError("teacher_conversations are required for teacher forcing")
        contexts = [format_endochat_teacher_context(sample) for sample in conversations]
        return ([item[0] for item in contexts], [item[1] for item in contexts])

    @torch.no_grad()
    def _teacher_target_logits(self, full_texts, prompt_texts, images_teacher):
        if bool(getattr(self.args, "distill_use_image_for_teacher", False)):
            if not isinstance(images_teacher, torch.Tensor):
                raise RuntimeError(
                    "multimodal teacher distillation requires a stacked images_teacher tensor"
                )
            teacher_dtype = next(self.teacher.parameters()).dtype
            images_teacher = images_teacher.to(dtype=teacher_dtype)
        else:
            images_teacher = None
        full_token_ids = [
            self.teacher_tokenizer.encode(text, bos=True, eos=False) for text in full_texts
        ]
        prompt_token_ids = [
            self.teacher_tokenizer.encode(text, bos=True, eos=False) for text in prompt_texts
        ]
        raw_logits = self.teacher.compute_logits(
            full_token_ids, images=images_teacher, bos=False, eos=False
        )
        target_logits = []
        target_token_ids = []
        for (full_ids, prompt_ids, logits) in zip(full_token_ids, prompt_token_ids, raw_logits):
            target_start = _common_prefix_length(full_ids, prompt_ids)
            first_logit = max(0, target_start - 1)
            last_logit = max(first_logit + 1, len(full_ids) - 1)
            selected = logits[first_logit:last_logit]
            if selected.size(0) == 0:
                selected = logits[-1:].clone()
            selected_ids = full_ids[target_start:]
            if selected.size(0) != len(selected_ids):
                raise RuntimeError(
                    f"teacher answer logits and token ids are not position-aligned: logits={selected.size(0)} ids={len(selected_ids)}"
                )
            target_logits.append(selected.float())
            target_token_ids.append(selected_ids)
        return (target_logits, target_token_ids)

    @staticmethod
    def _masked_mean_per_sample(values, mask):
        values = values * mask.to(values.dtype)
        denominator = mask.sum(dim=1).clamp_min(1).to(values.dtype)
        return values.sum(dim=1) / denominator

    def _pack_student_targets(self, values, target_mask):
        packed = []
        for (sample, mask) in zip(values, target_mask):
            packed.append(sample[mask])
        return packed
