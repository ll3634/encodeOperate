"""Train Probe-S (sufficiency = baseline EM) and compare with Probe-R (relevance = 0-doc vs 1+-doc)."""
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from collections import Counter

SEP = "=" * 60

# ── Load activations ──
d = np.load('results/phase1_probe/activations_multilayer.npz', allow_pickle=True)
X_l20 = d['layer_20']
sample_ids = list(d['sample_ids'])
y_relevance = d['y']
print(f"Activations: {X_l20.shape}, labels (R): {y_relevance.shape}")
print(f"  Probe-R label dist: 0={int((y_relevance==0).sum())}, 1={int((y_relevance==1).sum())}")

# ── Load baseline EM labels ──
baseline = {}
with open('results/l20_rho020_n500/baseline_results.jsonl') as f:
    for line in f:
        r = json.loads(line)
        baseline[r['sample_id']] = int(r.get('em_correct', r.get('is_correct', False)))

y_sufficiency = np.array([baseline.get(sid, -1) for sid in sample_ids], dtype=np.int32)
valid_mask = y_sufficiency >= 0
X_valid = X_l20[valid_mask]
y_s = y_sufficiency[valid_mask]
y_r = y_relevance[valid_mask]
print(f"Matched: {valid_mask.sum()}/{len(sample_ids)}")
print(f"  Probe-S label dist: 0(wrong)={int((y_s==0).sum())}, 1(correct)={int((y_s==1).sum())}")

# ── Cross-tab ──
ct = Counter(zip(y_r.tolist(), y_s.tolist()))
print(f"\n{SEP}\nLABEL CROSS-TAB\n{SEP}")
print(f"  R=0,S=0 (0-doc & wrong):    {ct.get((0,0),0)}")
print(f"  R=0,S=1 (0-doc & correct):  {ct.get((0,1),0)}")
print(f"  R=1,S=0 (1+-doc & wrong):   {ct.get((1,0),0)}")
print(f"  R=1,S=1 (1+-doc & correct): {ct.get((1,1),0)}")

# ── Train Probe-R ──
scaler_r = StandardScaler()
X_r_scaled = scaler_r.fit_transform(X_valid)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores_r = cross_val_score(
    LogisticRegression(class_weight='balanced', C=1.0, max_iter=2000, solver='lbfgs', random_state=42),
    X_r_scaled, y_r, cv=cv, scoring='roc_auc')
print(f"\nProbe-R 5-fold CV AUROC: {scores_r.mean():.3f} +/- {scores_r.std():.3f}")

clf_r = LogisticRegression(class_weight='balanced', C=1.0, max_iter=2000, solver='lbfgs', random_state=42)
clf_r.fit(X_r_scaled, y_r)
w_r = clf_r.coef_[0] / scaler_r.scale_
dir_r = (w_r / np.linalg.norm(w_r)).astype(np.float64)

# ── Train Probe-S ──
scaler_s = StandardScaler()
X_s_scaled = scaler_s.fit_transform(X_valid)

scores_s = cross_val_score(
    LogisticRegression(class_weight='balanced', C=1.0, max_iter=2000, solver='lbfgs', random_state=42),
    X_s_scaled, y_s, cv=cv, scoring='roc_auc')
print(f"Probe-S 5-fold CV AUROC: {scores_s.mean():.3f} +/- {scores_s.std():.3f}")

scores_s_ba = cross_val_score(
    LogisticRegression(class_weight='balanced', C=1.0, max_iter=2000, solver='lbfgs', random_state=42),
    X_s_scaled, y_s, cv=cv, scoring='balanced_accuracy')
print(f"Probe-S 5-fold CV BalAcc: {scores_s_ba.mean():.3f} +/- {scores_s_ba.std():.3f}")

clf_s = LogisticRegression(class_weight='balanced', C=1.0, max_iter=2000, solver='lbfgs', random_state=42)
clf_s.fit(X_s_scaled, y_s)
w_s = clf_s.coef_[0] / scaler_s.scale_
dir_s = (w_s / np.linalg.norm(w_s)).astype(np.float64)

# ── Load existing directions ──
dir_r_orig = np.load('results/phase1_probe/probe_direction_l20.npz')['decision_direction'].astype(np.float64)
dir_r_orig /= np.linalg.norm(dir_r_orig)

dir_action = np.load('steering/directions/direction_search_v3_layer20.npz')['decision_direction'].astype(np.float64)
dir_action /= np.linalg.norm(dir_action)

# ── Cosines ──
print(f"\n{SEP}\nCOSINE SIMILARITIES\n{SEP}")
pairs = [
    ("Probe-R(retrained)", "Probe-S", dir_r, dir_s),
    ("Probe-R(retrained)", "action_dir", dir_r, dir_action),
    ("Probe-S", "action_dir", dir_s, dir_action),
    ("Probe-R(orig)", "Probe-R(retrained)", dir_r_orig, dir_r),
    ("Probe-R(orig)", "Probe-S", dir_r_orig, dir_s),
    ("Probe-R(orig)", "action_dir", dir_r_orig, dir_action),
]
for n1, n2, d1, d2 in pairs:
    cos = float(np.dot(d1, d2))
    print(f"  cos({n1}, {n2}) = {cos:.4f}")

# ── Held-out test ──
print(f"\n{SEP}\nHELD-OUT TEST (80/20)\n{SEP}")
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

tr, te = next(sss.split(X_valid, y_s))
clf_s2 = LogisticRegression(class_weight='balanced', C=1.0, max_iter=2000, solver='lbfgs', random_state=42)
clf_s2.fit(X_s_scaled[tr], y_s[tr])
prob_s = clf_s2.predict_proba(X_s_scaled[te])[:,1]
print(f"Probe-S test AUROC: {roc_auc_score(y_s[te], prob_s):.3f}")
print(f"Probe-S test BalAcc: {balanced_accuracy_score(y_s[te], clf_s2.predict(X_s_scaled[te])):.3f}")

tr2, te2 = next(sss.split(X_valid, y_r))
clf_r2 = LogisticRegression(class_weight='balanced', C=1.0, max_iter=2000, solver='lbfgs', random_state=42)
clf_r2.fit(X_r_scaled[tr2], y_r[tr2])
prob_r = clf_r2.predict_proba(X_r_scaled[te2])[:,1]
print(f"Probe-R test AUROC: {roc_auc_score(y_r[te2], prob_r):.3f}")

# ── Angle in degrees ──
cos_rs = float(np.dot(dir_r, dir_s))
angle_deg = np.degrees(np.arccos(np.clip(abs(cos_rs), 0, 1)))
print(f"\nAngle between Probe-R and Probe-S: {angle_deg:.1f} degrees")
print(f"|cos| = {abs(cos_rs):.4f}")
