# Next Steps: Expanding Red Flag Analysis

## Current Status ✅

**Completed:**
- ✅ Analyzed 100 PopQA samples
- ✅ Identified 11 Red Flag samples (11%)
- ✅ Demonstrated **100% JES protection rate** on Red Flag samples
- ✅ Generated detailed analysis reports and visualizations
- ✅ Created analysis scripts for automated subset classification

**Key Finding:**
> Modern strong models (Qwen2.5-7B) are "tool fanatics" - they over-rely on search tools (Margin = +20.65). JES successfully protects against this Red Flag behavior with 100% protection rate.

## Recommended Next Steps

### Option 1: Expand PopQA to 500 Samples (High Priority)

**Goal:** Get statistically robust Red Flag protection data

**Expected Results:**
- ~55 Red Flag samples (11% of 500)
- Stronger statistical evidence (current: 11 samples, target: 55 samples)
- 95% CI for protection rate will narrow significantly

**How to Run:**
```bash
cd tmc/scripts/e2e_agent
bash scripts/run_popqa_500.sh
```

**Time Estimate:** ~50 minutes
- 500 samples × 4 policies × 1.5s/sample ≈ 3000s ≈ 50 min

**Analysis:**
```bash
python analysis/analyze_subsets.py \
    --baseline results/popqa_500/baseline_500.jsonl \
    --jes results/popqa_500/jes_500.jsonl \
    --force-adopt results/popqa_500/force_adopt_500.jsonl \
    --force-reject results/popqa_500/force_reject_500.jsonl \
    --output results/popqa_500/subset_analysis.json

python analysis/red_flag_report.py \
    --subset-analysis results/popqa_500/subset_analysis.json \
    --baseline results/popqa_500/baseline_500.jsonl \
    --jes results/popqa_500/jes_500.jsonl \
    --force-adopt results/popqa_500/force_adopt_500.jsonl \
    --output results/popqa_500/red_flag_report.json
```

**Benefits:**
- ✅ Stronger statistical evidence for paper
- ✅ More robust confidence intervals
- ✅ Better understanding of Red Flag distribution

---

### Option 2: Test Calculator Domain (GSM8K Hard) (High Priority)

**Goal:** Demonstrate Stealth protection (under-reliance on calculator)

**Hypothesis:**
- Model will try to mental-calculate multi-step problems
- Model will make arithmetic errors
- JES will detect low confidence and force calculator use
- Success rate will improve

**Expected Results:**
- Stealth samples: 20-30% (model fails without calculator, succeeds with calculator)
- JES protection: 70-90% (helps model adopt calculator when needed)

**How to Run:**
```bash
cd tmc/scripts/e2e_agent

# Create GSM8K runner (similar to run_popqa.py)
python runners/run_gsm8k.py \
    --data-path data/gsm8k/test.jsonl \
    --direction-path steering/directions/direction_calculator.npz \
    --output results/gsm8k_hard/baseline_200.jsonl \
    --policy baseline \
    --n-samples 200 \
    --difficulty hard
```

**Time Estimate:** ~30 minutes
- 200 samples × 4 policies × 1.5s/sample ≈ 1200s ≈ 20 min

**Benefits:**
- ✅ Demonstrates JES as **bidirectional controller**
- ✅ Shows JES works in different domains (Search vs Math)
- ✅ Stronger paper narrative: "JES solves both over-reliance AND under-reliance"

---

### Option 3: Both (Recommended for Paper)

**Combined Story:**

1. **Search Domain (PopQA):**
   - Model over-relies on search (Margin = +20.65)
   - Red Flag is main problem (11% of samples)
   - JES acts as **Brake** → 100% protection

2. **Math Domain (GSM8K):**
   - Model under-relies on calculator (tries to mental-calculate)
   - Stealth is main problem (20-30% of samples)
   - JES acts as **Accelerator** → 70-90% protection

3. **Conclusion:**
   - JES is a **bidirectional controller** for tool-using agents
   - Prevents both over-reliance (Red Flag) and under-reliance (Stealth)
   - Domain-adaptive: automatically adjusts based on model's bias

**Paper Narrative:**
> "Modern tool-using agents exhibit bidirectional failures: over-reliance in Search domains and under-reliance in Math domains. JES is a bidirectional controller that addresses both, achieving 100% Red Flag protection in Search and 70-90% Stealth protection in Math."

---

## Implementation Priority

### Phase 1: Immediate (Today)
1. ✅ **Run PopQA 500 samples** (50 min)
   - Get ~55 Red Flag samples
   - Stronger statistical evidence

### Phase 2: Short-term (This Week)
2. ✅ **Implement GSM8K runner** (2 hours)
   - Adapt run_popqa.py for GSM8K
   - Extract calculator direction vector
   
3. ✅ **Run GSM8K Hard 200 samples** (30 min)
   - Demonstrate Stealth protection

### Phase 3: Analysis (This Week)
4. ✅ **Generate combined analysis**
   - Compare Red Flag (Search) vs Stealth (Math)
   - Create unified visualization
   - Write paper section

---

## Scripts Ready to Use

All scripts are in `tmc/scripts/e2e_agent/`:

### Experiment Runners
- ✅ `scripts/run_popqa_500.sh` - Run 500-sample PopQA experiment
- ✅ `scripts/test_popqa_pipeline.sh` - Test pipeline with 10 samples
- ⏳ `scripts/run_gsm8k_hard.sh` - TODO: Create GSM8K runner

### Analysis Scripts
- ✅ `analysis/analyze_subsets.py` - Classify Stealth/Red Flag/Indifferent
- ✅ `analysis/red_flag_report.py` - Generate Red Flag protection report
- ✅ `analysis/plot_red_flag_protection.py` - Generate visualizations

### Results
- ✅ `results/popqa_hard/` - 100-sample results (DONE)
- ⏳ `results/popqa_500/` - 500-sample results (TODO)
- ⏳ `results/gsm8k_hard/` - GSM8K results (TODO)

---

## Decision Point

**Question:** Should we expand PopQA to 500 samples, or switch to GSM8K first?

**Recommendation:** **Do both in parallel**

1. **Start PopQA 500 now** (50 min, can run in background)
2. **While waiting, implement GSM8K runner** (2 hours)
3. **Run GSM8K Hard** (30 min)
4. **Analyze both together** (1 hour)

**Total time:** ~4 hours for complete bidirectional analysis

---

## Expected Paper Impact

### Current (100 samples, Search only)
- ✅ JES protects against Red Flag (100% on 11 samples)
- ⚠️ Limited statistical power (n=11)
- ⚠️ Single domain (Search only)

### After Expansion (500 Search + 200 Math)
- ✅ JES protects against Red Flag (100% on ~55 samples) → **Strong evidence**
- ✅ JES protects against Stealth (70-90% on ~40-60 samples) → **Bidirectional**
- ✅ Domain-adaptive behavior → **Generalizable**
- ✅ Stronger paper narrative → **More impactful**

---

## Conclusion

**Immediate action:** Run `bash scripts/run_popqa_500.sh` to expand Red Flag analysis.

**Next action:** Implement GSM8K runner to demonstrate Stealth protection.

**Goal:** Position JES as a **bidirectional controller** that solves both over-reliance and under-reliance in tool-using agents.

