"""Quick diagnostic: are PopQA logits NaN with vs without early-layer hooks?"""
import sys, os, json, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from agent.prompts import PromptBuilder
from steering.hook_utils import get_model_layers
from scripts.cross_model_full import apply_chat_template_safe, compute_margin

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
print(f"loading {MODEL}")
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
m.eval()
layers = get_model_layers(m)
n_layers = len(layers)
print(f"  n_layers={n_layers}")

samples = [json.loads(l) for l in open("data/popqa/popqa_test.jsonl")]
random.seed(42); random.shuffle(samples)
samples = samples[:5]
pb = PromptBuilder(tools=["search"])

def run(layer_set):
    captured = {}
    handles = []
    for li in layer_set:
        def make(li_=li):
            def h(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured.setdefault(li_, []).append(h[0,-1,:].detach().float().cpu().numpy())
            return h
        handles.append(layers[li].register_forward_hook(make()))
    margins = []
    logit_minmax = []
    for s in samples:
        msgs = pb.build_full_prompt(s["question"], [])
        prompt = apply_chat_template_safe(tok, msgs)
        ids = tok.encode(prompt, return_tensors="pt").to(next(m.parameters()).device)
        with torch.no_grad():
            lg = m(ids).logits
        last = lg[0,-1,:]
        margins.append(compute_margin(last, tok))
        logit_minmax.append((float(last.float().min()), float(last.float().max()), bool(torch.isnan(last).any())))
    for h in handles: h.remove()
    return margins, logit_minmax, captured

print("\n--- NO HOOKS ---")
mm, lm, _ = run([])
for i, (mar, (lo, hi, nan)) in enumerate(zip(mm, lm)):
    print(f"  s{i}: margin={mar:.3f}  logit=[{lo:.2f},{hi:.2f}]  nan={nan}")

print("\n--- HOOKS ON LATE LAYERS [16,20,24,28] ---")
mm, lm, cap = run([16, 20, 24, 28])
for i, (mar, (lo, hi, nan)) in enumerate(zip(mm, lm)):
    print(f"  s{i}: margin={mar:.3f}  logit=[{lo:.2f},{hi:.2f}]  nan={nan}")
for li, arrs in cap.items():
    arr = np.stack(arrs)
    print(f"  cap L{li}: shape={arr.shape}  finite={np.isfinite(arr).all()}  norm_mean={np.linalg.norm(arr,axis=1).mean():.2f}")

print("\n--- HOOKS ON ALL SWEEP [4,6,8,10,12,14,16,18,20,22,24,26,28,30] ---")
mm, lm, cap = run([4,6,8,10,12,14,16,18,20,22,24,26,28,30])
for i, (mar, (lo, hi, nan)) in enumerate(zip(mm, lm)):
    print(f"  s{i}: margin={mar:.3f}  logit=[{lo:.2f},{hi:.2f}]  nan={nan}")
for li, arrs in cap.items():
    arr = np.stack(arrs)
    print(f"  cap L{li}: shape={arr.shape}  finite={np.isfinite(arr).all()}  norm_mean={np.linalg.norm(arr,axis=1).mean():.2f}")
