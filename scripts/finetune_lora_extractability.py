#!/usr/bin/env python3
"""LoRA SFT pilot for the extractability fine-tune experiment.

Trains a single LoRA adapter on (prompt -> target) pairs from
data/extractability_train/train_{T0,N0}.jsonl. Loss is masked so only the
assistant target tokens contribute to CE.

Usage:
  python scripts/finetune_lora_extractability.py \
      --train-data data/extractability_train/train_T0.jsonl \
      --output-dir adapters/qwen_t0_v1
"""
import argparse, json, math, random, time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_cosine_schedule_with_warmup)
from peft import LoraConfig, get_peft_model


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=1500):
        self.tok = tokenizer
        self.max_len = max_len
        self.recs = []
        n_drop = 0
        for l in open(path):
            r = json.loads(l)
            prompt_str = tokenizer.apply_chat_template(
                r["prompt_messages"], tokenize=False, add_generation_prompt=True)
            full_str = prompt_str + r["target_text"] + tokenizer.eos_token
            prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_str, add_special_tokens=False)["input_ids"]
            if len(full_ids) > max_len:
                n_drop += 1
                continue
            r["_full_ids"] = full_ids
            r["_n_prompt"] = len(prompt_ids)
            self.recs.append(r)
        print(f"[SFTDataset] kept={len(self.recs)} dropped={n_drop} (max_len={max_len})")

    def __len__(self): return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        full_ids = r["_full_ids"]
        n_prompt = r["_n_prompt"]
        labels = [-100] * n_prompt + full_ids[n_prompt:]
        return {"input_ids": full_ids, "labels": labels,
                "attention_mask": [1] * len(full_ids)}


def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        pad = L - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * pad)
        out["labels"].append(b["labels"] + [-100] * pad)
        out["attention_mask"].append(b["attention_mask"] + [0] * pad)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-grad-norm", type=float, default=0.3)
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="Disable gradient checkpointing (needs ~50 GB; default is enabled).")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                    help="Base model precision. NOTE: fp32 path currently produces NaN logits "
                         "with Qwen2.5 + transformers 4.45 (separate upstream bug); use bf16.")
    ap.add_argument("--logit-clamp", type=float, default=50.0,
                    help="Symmetric clamp applied to fp32-cast logits before CE. "
                         "Guards against bf16 activation overflow on rare batches.")
    ap.add_argument("--diag", action="store_true",
                    help="Per-microbatch logit_max / nan trace.")
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out_dir / "train_args.json", "w"), indent=2)

    print(f"[load] tokenizer + model {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    base_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=base_dtype,
        device_map="cuda:0", trust_remote_code=True,
        attn_implementation="eager")
    model.config.use_cache = False

    lcfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    # Print actual dtypes of trainable params for debugging.
    sample_dtypes = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            sample_dtypes.setdefault(p.dtype, n)
    print(f"[dtype] trainable param dtypes: {dict((str(k),v) for k,v in sample_dtypes.items())}")

    ds = SFTDataset(args.train_data, tok, max_len=args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    n_steps = math.ceil(len(dl) / args.grad_accum) * args.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    sch = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=int(args.warmup_ratio * n_steps),
        num_training_steps=n_steps)
    print(f"[train] n_records={len(ds)} steps={n_steps} epochs={args.epochs}")

    def compute_loss_fp32(batch):
        # Forward without internal loss; cast logits to fp32 for stable CE.
        labels = batch.pop("labels")
        out = model(**batch, labels=None)
        logits = out.logits.float()
        if args.logit_clamp > 0:
            logits = logits.clamp(-args.logit_clamp, args.logit_clamp)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss, logits

    model.train()
    log = []
    step = 0; t0 = time.time(); accum_loss = 0.0; micro = 0; nan_streak = 0
    skipped_steps = 0
    trainable = [p for p in model.parameters() if p.requires_grad]
    for ep in range(args.epochs):
        for batch in dl:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            sl_dbg = batch["input_ids"].shape[1]
            loss, logits_dbg = compute_loss_fp32(batch)
            if args.diag:
                lmax = float(logits_dbg.detach().abs().max().item())
                print(f"  micro={micro} seq_len={sl_dbg} loss={loss.item():.4f} logit_abs_max={lmax:.2e}")
            if not torch.isfinite(loss):
                opt.zero_grad()
                nan_streak += 1
                print(f"  [warn] non-finite loss at micro={micro} step={step} "
                      f"(streak={nan_streak}); skipping batch")
                if nan_streak >= 5:
                    raise RuntimeError("5 consecutive non-finite losses; aborting.")
                continue
            nan_streak = 0
            (loss / args.grad_accum).backward()
            accum_loss += loss.item()
            micro += 1
            if micro % args.grad_accum == 0:
                # Pre-step grad sanity check: bf16 backward can produce NaN/Inf
                # in trainable LoRA params even when loss looked finite.
                bad = False
                for p in trainable:
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        bad = True; break
                if bad:
                    opt.zero_grad(); skipped_steps += 1
                    print(f"  [warn] non-finite grad at step boundary "
                          f"micro={micro} step={step} (skipped={skipped_steps}); skipping opt.step")
                    accum_loss = 0.0
                    if skipped_steps >= 5:
                        raise RuntimeError("5 consecutive non-finite grad steps; aborting.")
                    continue
                gnorm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                if not torch.isfinite(gnorm):
                    opt.zero_grad(); skipped_steps += 1
                    print(f"  [warn] non-finite grad_norm={float(gnorm)} step={step}; skipping opt.step")
                    accum_loss = 0.0
                    continue
                skipped_steps = 0
                opt.step(); sch.step(); opt.zero_grad()
                step += 1
                avg = accum_loss / args.grad_accum
                log.append({"step": step, "epoch": ep, "loss": avg,
                            "lr": sch.get_last_lr()[0],
                            "grad_norm": float(gnorm)})
                if step <= 3 or step % 5 == 0 or step == n_steps:
                    print(f"  step {step}/{n_steps} ep={ep} "
                          f"loss={avg:.4f} grad={float(gnorm):.3f} "
                          f"lr={sch.get_last_lr()[0]:.2e} "
                          f"elapsed={time.time()-t0:.1f}s")
                accum_loss = 0.0
    print(f"[save] adapter -> {out_dir}")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    json.dump(log, open(out_dir / "train_log.json", "w"), indent=2)
    print(f"[done] {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
