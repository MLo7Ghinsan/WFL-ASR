from copy import deepcopy
from pathlib import Path
import torch
import yaml

def read_checkpoint(path):
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("expected a WFL .ckpt with 'state_dict' bitch like honestly idk how it's missing that")
    return checkpoint

def deep_merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base

def load_wfl_config(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f)
    ft = user_cfg.get("finetune", {})
    if not ft.get("enabled", False):
        return user_cfg

    source = ft.get("checkpoint_path")
    if not source:
        raise ValueError("finetune.checkpoint_path is required")
    source_path = Path(source).expanduser()
    if not source_path.is_absolute():
        source_path = path.parent / source_path
    source_path = source_path.resolve()

    source_ckpt = read_checkpoint(source_path)
    hyper = source_ckpt.get("hyper_parameters", {})
    source_cfg = hyper.get("config")
    if not isinstance(source_cfg, dict):
        sibling_yaml = source_path.parent / "config.yaml"
        if not sibling_yaml.is_file():
            raise ValueError("checkpoint has no config in hyper_parameters and no config.yaml along with it")
        with sibling_yaml.open(encoding="utf-8") as f:
            source_cfg = yaml.safe_load(f)

    cfg = deep_merge(deepcopy(source_cfg), user_cfg)
    cfg.setdefault("finetune", {})["checkpoint_path"] = str(source_path)
    cfg.setdefault("model", {})["freeze_encoder"] = True
    cfg["model"]["unfreeze_last_n_layers"] = 0

    if "save_dir" not in user_cfg.get("output", {}):
        raise ValueError("finetuning must define a NEW output.save_dir.")
    if Path(cfg["output"]["save_dir"]).expanduser().resolve() == source_path.parent:
        raise ValueError("fine-tuning output.save_dir cannot be the source checkpoint folder.")
    return cfg

def _langs_from_file(path):
    result = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            name, idx = line.strip().rsplit(",", 1)
            result[name] = int(idx)
    return result

def transfer_wfl_weights(model, source_ckpt, source_path, target_dir, target_labels):
    source_path = Path(source_path)
    source = source_ckpt["state_dict"]
    target = model.state_dict()
    special = {
        "model.classifier.weight", "model.classifier.bias",
        "model.lang_emb.weight",
    }

    # this transfer expects the WFL architecture to come from the source checkpoint
    # only language count and vocabulary dimensions can change
    for key, tensor in target.items():
        if key in special:
            continue
        if key not in source:
            raise ValueError(f"missing checkpoint parameter: {key}")
        if source[key].shape != tensor.shape:
            raise ValueError(
                f"incompatible layer {key}: source {tuple(source[key].shape)} "
                f"vs target {tuple(tensor.shape)}"
            )
        target[key] = source[key]

    hp = source_ckpt.get("hyper_parameters", {})
    old_labels = source_ckpt.get("wfl_labels") or hp.get("label_list")
    if old_labels is None:
        label_file = source_path.parent / "phonemes.txt"
        if not label_file.exists():
            raise ValueError("Cannot remap classifier: source labels unavailable.")
        old_labels = [x.strip() for x in label_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    old_labels = list(old_labels)
    if len(old_labels) != source["model.classifier.weight"].shape[0]:
        raise ValueError("Source label list does not match classifier rows.")
    old_label_to_row = {label: i for i, label in enumerate(old_labels)}

    matched = 0
    for new_idx, label in enumerate(target_labels):
        old_idx = old_label_to_row.get(label)
        if old_idx is not None:
            target["model.classifier.weight"][new_idx] = source["model.classifier.weight"][old_idx]
            target["model.classifier.bias"][new_idx] = source["model.classifier.bias"][old_idx]
            matched += 1

    old_langs = source_ckpt.get("wfl_lang2id")
    if old_langs is None:
        langs_file = source_path.parent / "langs.txt"
        old_langs = _langs_from_file(langs_file) if langs_file.exists() else {}
    new_langs_file = Path(target_dir) / "langs.txt"
    if not new_langs_file.exists():
        raise FileNotFoundError(f"missing target language map: {new_langs_file}")
    new_langs = _langs_from_file(new_langs_file)
    old_mat = source["model.lang_emb.weight"]
    reused_langs = []
    for lang, new_idx in new_langs.items():
        old_idx = old_langs.get(lang)
        if old_idx is not None and 0 <= old_idx < len(old_mat):
            target["model.lang_emb.weight"][new_idx] = old_mat[old_idx]
            reused_langs.append(lang)

    model.load_state_dict(target, strict=True)
    print(f"transferred {matched}/{len(target_labels)} classifier labels by name.")
    print(f"reused language embeddings: {reused_langs}")
    print(f"new or unmapped languages: {sorted(set(new_langs) - set(reused_langs))}")
    print("all shared acoustic layers transferred")
