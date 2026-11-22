"""Model loading, eligible-parameter discovery, prompt building and batched greedy generation."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
from torch import nn

from sfcd.config import ModelCfg
from sfcd.tasks.base import Task

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast(cfg: ModelCfg, device: torch.device):
    dt = _DTYPES[cfg.compute_dtype]
    if device.type != "cuda" or dt == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=dt)


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(cfg: ModelCfg, device: torch.device | None = None):
    """Load in fp32 master weights (compute happens under autocast)."""
    from transformers import AutoModelForCausalLM

    device = device or get_device()
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.name, torch_dtype=torch.float32)
    except ValueError:  # multimodal checkpoints (e.g. *ForConditionalGeneration)
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(cfg.name, torch_dtype=torch.float32)
    model.to(device)
    model.config.use_cache = False
    return model, load_tokenizer(cfg.name)


# ----------------------------------------------------------------------------- eligible params


@dataclass(frozen=True)
class LinearInfo:
    name: str  # module name, e.g. "model.layers.3.mlp.up_proj"
    layer: int  # transformer block index
    out_features: int
    in_features: int

    @property
    def numel(self) -> int:
        return self.out_features * self.in_features

    @property
    def param_name(self) -> str:
        return f"{self.name}.weight"


_SKIP = ("visual", "vision", "audio", "mtp")


def find_decoder_layers(model: nn.Module) -> tuple[str, nn.ModuleList]:
    """The longest ModuleList named '*layers' outside any vision/audio tower."""
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and name.split(".")[-1] in ("layers", "h", "blocks"):
            if any(s in name.lower() for s in _SKIP):
                continue
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    if best is None:
        raise RuntimeError("could not find transformer decoder layers")
    return best


def eligible_linears(model: nn.Module) -> list[LinearInfo]:
    """All nn.Linear weights inside transformer blocks (embeddings, norms, lm_head excluded)."""
    prefix, layers = find_decoder_layers(model)
    out = []
    for i, layer in enumerate(layers):
        for sub, mod in layer.named_modules():
            if isinstance(mod, nn.Linear):
                out.append(LinearInfo(f"{prefix}.{i}.{sub}", i, mod.out_features, mod.in_features))
    return out


def n_layers(model: nn.Module) -> int:
    return len(find_decoder_layers(model)[1])


# ----------------------------------------------------------------------------- prompts


def build_prompt(tok, task: Task, context_text: str | None, query: str) -> str:
    messages = [{"role": "system", "content": task.system_prompt},
                {"role": "user", "content": task.user_message(context_text, query)}]
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    return f"System: {messages[0]['content']}\nUser: {messages[1]['content']}\nAssistant:"


@torch.no_grad()
def stop_token_ids(model, tok) -> list[int]:
    """End-of-turn ids. Qwen's chat turn ends with <|im_end|> (tok.eos_token), but the model config's eos
    is <|endoftext|>; without both, fine-tuned students can run past <|im_end|> ("stonestonestone")."""
    ids = []
    gc = getattr(model, "generation_config", None)
    cfg_eos = getattr(gc, "eos_token_id", None)
    for x in [tok.eos_token_id, *(cfg_eos if isinstance(cfg_eos, list) else [cfg_eos])]:
        if x is not None and x not in ids:
            ids.append(x)
    for t in ("<|im_end|>", "<|endoftext|>"):
        x = tok.convert_tokens_to_ids(t)
        if isinstance(x, int) and x != tok.unk_token_id and x not in ids:
            ids.append(x)
    return ids


def generate(model, tok, prompts: list[str], max_new_tokens: int, batch_size: int,
             model_cfg: ModelCfg) -> list[str]:
    """Greedy batched generation; returns only the new text. Prompts are length-sorted for efficiency."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    tok.padding_side = "left"
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs: list[str] = [""] * len(prompts)
    stops = stop_token_ids(model, tok)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with autocast(model_cfg, device):
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=stops, use_cache=True)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            outs[i] = t.strip()
    model.train(was_training)
    return outs
