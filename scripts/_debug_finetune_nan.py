#!/usr/bin/env python3
"""Minimal NaN diagnostic for the LoRA SFT pipeline.

Loads ONE batch, runs forward pass through 4 progressively complex configs:
  (1) base bf16, no LoRA, no ckpt
  (2) base fp32, no LoRA, no ckpt
  (3) base bf16 + LoRA, no ckpt
  (4) base bf16 + LoRA + grad ckpt

For each: print logit max/min/mean, count NaNs, compute CE loss.
Also dumps the first batch's input/label structure to detect data bugs.
"""
import sys, json, torch
sys.path.insert(0, "scripts")
from finetune_lora_extractability import SFTDataset, collate
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATA = "data/extractability_train/train_T0.jsonl"


def get_first_batch():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    torch.manual_seed(42)
    ds = SFTDataset(DATA, tok, max_len=1024)
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=2, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    batch = next(iter(dl))
    return tok, batch


def inspect_batch(batch, tok):
    ids = batch["input_ids"]
    am = batch["attention_mask"]
    lb = batch["labels"]
    print(f"  input_ids shape={ids.shape} min={ids.min().item()} max={ids.max().item()} vocab={tok.vocab_size}")
    print(f"  attention_mask sum/seq: {am.sum(-1).tolist()}  (should equal real lengths)")
    print(f"  labels valid (non -100) per row: {(lb != -100).sum(-1).tolist()}")
    print(f"  labels max non-pad: {lb[lb != -100].max().item()} min: {lb[lb != -100].min().item()}")
    # OOV check
    if ids.max().item() >= tok.vocab_size:
        # could still be valid if it's a special token added
        print(f"  WARN: max input_id={ids.max().item()} >= vocab_size={tok.vocab_size} (added tokens?)")


def forward_check(model, batch, tag):
    batch_d = {k: v.to(model.device) for k, v in batch.items()}
    labels = batch_d.pop("labels")
    with torch.no_grad():
        out = model(**batch_d)
    logits = out.logits.float()
    nan = (~torch.isfinite(logits)).sum().item()
    lmax, lmin, lmean = logits.max().item(), logits.min().item(), logits.mean().item()
    # compute CE on label-valid positions
    sl = logits[..., :-1, :].contiguous()
    slab = labels[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        sl.view(-1, sl.size(-1)), slab.view(-1), ignore_index=-100)
    print(f"[{tag}] logits nan={nan} max={lmax:.2e} min={lmin:.2e} mean={lmean:.3e} loss={loss.item():.4f}")
    return loss.item()


def make_lora(model):
    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"])
    return get_peft_model(model, lcfg)


def run(dtype, with_lora, with_ckpt, tag, batch):
    print(f"\n=== {tag} ===")
    free_before = torch.cuda.mem_get_info(0)[0] / 1e9
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=dtype, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="eager")
    m.config.use_cache = False
    if with_lora:
        m = make_lora(m)
    if with_ckpt:
        m.gradient_checkpointing_enable()
        if hasattr(m, "enable_input_require_grads"):
            m.enable_input_require_grads()
    m.eval()
    free_after = torch.cuda.mem_get_info(0)[0] / 1e9
    print(f"  loaded; gpu free {free_before:.1f}->{free_after:.1f} GiB")
    forward_check(m, batch, tag)
    del m
    torch.cuda.empty_cache()


def main():
    tok, batch = get_first_batch()
    print("\n=== batch 1 inspection ===")
    inspect_batch(batch, tok)
    # Configs ordered cheapest first to fail fast
    run(torch.bfloat16, False, False, "bf16 base only", batch)
    run(torch.bfloat16, True,  False, "bf16 + LoRA", batch)
    run(torch.bfloat16, True,  True,  "bf16 + LoRA + ckpt", batch)
    run(torch.float32, False, False, "fp32 base only", batch)
    run(torch.float32, True,  False, "fp32 + LoRA", batch)


if __name__ == "__main__":
    main()
