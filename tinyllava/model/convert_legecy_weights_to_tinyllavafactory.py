import os
import json
import argparse
from huggingface_hub import hf_hub_download
import torch
from safetensors import safe_open
from .modeling_tinyllava import TinyLlavaForConditionalGeneration
from .configuration_tinyllava import TinyLlavaConfig

KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.vision_tower": "vision_tower._vision_tower",
    "model.mm_projector": "connector._connector",
    "model.embed_tokens": "language_model.model.embed_tokens",
    "model.layers": "language_model.model.layers",
    "model.norm": "language_model.model.norm",
    "lm_head": "language_model.lm_head",
    "model.final_layernorm": "language_model.model.final_layernorm",
}
KEYS_TO_MODELNAME_MAPPING = {
    "TinyLlavaLlamaForCausalLM": "TinyLlama/TinyLlama-1.1B-chat-v1.0",
    "TinyLlavaStablelmForCausalLM": "stabilityai/stablelm-2-zephyr-1_6b",
    "TinyLlavaPhiForCausalLM": "microsoft/phi-2",
    "bczhou/TinyLLaVA-3.1B-SigLIP": "google/siglip-so400m-patch14-384",
    "bczhou/TinyLLaVA-2.0B-SigLIP": "google/siglip-so400m-patch14-384",
    "bczhou/TinyLLaVA-1.5B-SigLIP": "google/siglip-so400m-patch14-384",
}
IGNORABLE_MISSING_PREFIXES = (
    "vision_tower._vision_tower.vision_model.encoder.layers.26.",
    "vision_tower._vision_tower.vision_model.head.",
    "grounding_head.",
)


def convert_legecy_config_to_tinyllavaconfig(old_config_path):
    if os.path.exists(old_config_path):
        config_path = os.path.join(old_config_path, "config.json")
    else:
        config_path = hf_hub_download(old_config_path, "config.json")
    with open(config_path, "r") as f:
        old_config = json.load(f)
    llm_model_name_or_path = KEYS_TO_MODELNAME_MAPPING[old_config["architectures"][0]]
    vision_model_name_or_path = KEYS_TO_MODELNAME_MAPPING[old_config["mm_vision_tower"]]
    model_config = TinyLlavaConfig(
        llm_model_name_or_path=llm_model_name_or_path,
        vision_model_name_or_path=vision_model_name_or_path,
        connector_type=old_config["mm_projector_type"],
        hidden_size=old_config["hidden_size"],
        vocab_size=old_config["vocab_size"],
        pad_token=old_config["pad_token"],
        tokenizer_padding_side=old_config["tokenizer_padding_side"],
        tokenizer_model_max_length=old_config["tokenizer_model_max_length"],
        vision_feature_layer=old_config["mm_vision_select_layer"],
        vision_feature_select_strategy=old_config["mm_vision_select_feature"],
        image_aspect_ratio=old_config["image_aspect_ratio"],
        use_cache=old_config["use_cache"],
    )
    return model_config


def convert_state_dict_to_tinyllavafactory(old_state_dict_path):
    old_state_dict = []
    if os.path.exists(old_state_dict_path):
        meta_file_name = os.path.join(old_state_dict_path, "model.safetensors.index.json")
        if os.path.exists(meta_file_name):
            with open(meta_file_name, "r") as f:
                meta_file = json.load(f)
            meta_file = list(set(meta_file["weight_map"].values()))
            for name in meta_file:
                old_state_dict.append(os.path.join(old_state_dict_path, name))
        else:
            old_state_dict.append(os.path.join(old_state_dict_path, "model.safetensors"))
    else:
        try:
            meta_file_name = hf_hub_download(old_state_dict_path, "model.safetensors.index.json")
            with open(meta_file_name, "r") as f:
                meta_file = json.load(f)
            meta_file = list(set(meta_file["weight_map"].values()))
            for name in meta_file:
                old_state_dict.append(hf_hub_download(old_state_dict_path, name))
        except:
            old_state_dict.append(hf_hub_download(old_state_dict_path, "model.safetensors"))
    state_dict = {}
    for osd in old_state_dict:
        with safe_open(osd, framework="pt", device="cpu") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
    return convert_legacy_state_dict_keys(state_dict)


def convert_legacy_state_dict_keys(state_dict):
    new_state_dict = {}
    for (key, value) in state_dict.items():
        for (key_to_modify, new_key) in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)
        new_state_dict[key] = value
    return new_state_dict


def unexpected_conversion_gaps(missing_keys, unexpected_keys):
    required_missing = [
        key for key in missing_keys if not key.startswith(IGNORABLE_MISSING_PREFIXES)
    ]
    return (required_missing, list(unexpected_keys))


def convert_legecy_weights_to_tinyllavafactory(
    old_state_dict_path, new_state_dict_path=None, torch_dtype=None
):
    model_config = convert_legecy_config_to_tinyllavaconfig(old_state_dict_path)
    model = TinyLlavaForConditionalGeneration(model_config)
    state_dict = convert_state_dict_to_tinyllavafactory(old_state_dict_path)
    incompatible = model.load_state_dict(state_dict, strict=False)
    (required_missing, unexpected) = unexpected_conversion_gaps(
        incompatible.missing_keys, incompatible.unexpected_keys
    )
    if required_missing or unexpected:
        raise RuntimeError(
            f"legacy checkpoint conversion was incomplete: missing={required_missing}, unexpected={unexpected}"
        )
    ignored = sorted(set(incompatible.missing_keys) - set(required_missing))
    if ignored:
        print(f"ignored {len(ignored)} documented legacy-only missing keys")
    del state_dict
    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)
        model.config.torch_dtype = torch_dtype
    if new_state_dict_path is not None:
        os.makedirs(new_state_dict_path, exist_ok=True)
        model.config.save_pretrained(new_state_dict_path)
        model.tokenizer.save_pretrained(new_state_dict_path)
        model.save_pretrained(new_state_dict_path, safe_serialization=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    model = convert_legecy_weights_to_tinyllavafactory(
        args.source, new_state_dict_path=args.output, torch_dtype=dtype
    )
    unique_parameters = sum((parameter.numel() for parameter in model.parameters()))
    print(
        json.dumps(
            {
                "source": os.path.realpath(args.source),
                "output": os.path.realpath(args.output),
                "dtype": args.dtype,
                "unique_parameters": unique_parameters,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
