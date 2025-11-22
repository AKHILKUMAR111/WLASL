import json
from typing import Dict, List, Any, Union

import torch
import numpy as np


def save_model_state(model: torch.nn.Module, path: str) -> None:
    """
    Save model state_dict to a .pth file.
    """
    torch.save(model.state_dict(), path)
    print(f"Saved model weights to {path}")


def id_to_gloss_from_gloss_to_id(gloss_to_id: Dict[str, int]) -> List[str]:
    """
    Convert a mapping {gloss: idx} to a list where list[idx] = gloss.
    """
    if not gloss_to_id:
        return []
    max_idx = max(gloss_to_id.values())
    mapping: List[str] = [""] * (max_idx + 1)
    for gloss, idx in gloss_to_id.items():
        mapping[idx] = gloss
    return mapping


def save_id_to_gloss_from_gloss_to_id(gloss_to_id: Dict[str, int], out_json: str) -> None:
    mapping = id_to_gloss_from_gloss_to_id(gloss_to_id)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"Saved id_to_gloss list (length {len(mapping)}) to {out_json}")


def save_id_to_gloss(mapping: List[str], out_json: str) -> None:
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"Saved id_to_gloss list (length {len(mapping)}) to {out_json}")


def load_id_to_gloss(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data: Any = json.load(f)
    if isinstance(data, dict):
        keys = sorted(data.keys(), key=lambda k: int(k))
        return [data[k] for k in keys]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported id_to_gloss format; expected list or {str(idx): gloss} dict")


# --- New helpers for id_to_gloss stored in NPZ as 0-D object array (dict) ---

def id_to_gloss_list_from_dict(id_to_gloss_dict: Dict[Union[int, str], str]) -> List[str]:
    """
    Convert an id->gloss dict (keys possibly str or int) into a list where
    list[idx] = gloss, ordered by idx.
    """
    if not id_to_gloss_dict:
        return []
    # Normalize keys to ints
    norm: Dict[int, str] = {}
    for k, v in id_to_gloss_dict.items():
        ki = int(k)  # handles both str and int keys
        norm[ki] = v
    max_idx = max(norm.keys())
    mapping: List[str] = [""] * (max_idx + 1)
    for idx, gloss in norm.items():
        mapping[idx] = gloss
    return mapping


def save_id_to_gloss_dict(id_to_gloss_dict: Dict[Union[int, str], str], out_json: str) -> None:
    """
    Save the raw dict form (keys kept as strings in JSON) for archival.
    """
    # Ensure JSON keys are strings
    json_ready = {str(k): v for k, v in id_to_gloss_dict.items()}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_ready, f, ensure_ascii=False, indent=2)
    print(f"Saved id_to_gloss dict with {len(json_ready)} entries to {out_json}")


def save_id_to_gloss_from_npz(npz_path: str, key: str, out_json_list: str, out_json_dict: str = "") -> None:
    """
    Load an NPZ file that contains an id_to_gloss object (0-D numpy array with a dict),
    convert it to a list (ordered by integer index), and save to JSON. Optionally saves
    the raw dict JSON as well.
    """
    data = np.load(npz_path, allow_pickle=True)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {npz_path}")
    id_to_gloss_obj = data[key].item()
    if not isinstance(id_to_gloss_obj, dict):
        raise ValueError(f"Expected a dict at key '{key}', got {type(id_to_gloss_obj)}")
    mapping_list = id_to_gloss_list_from_dict(id_to_gloss_obj)
    save_id_to_gloss(mapping_list, out_json_list)
    if out_json_dict:
        save_id_to_gloss_dict(id_to_gloss_obj, out_json_dict)
