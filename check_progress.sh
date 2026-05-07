#!/bin/bash
# Quick progress checker for 500-sample experiment

echo "========================================="
echo "PopQA 500 实验进度"
echo "========================================="
echo ""

# Check if process is running
if pgrep -f "run_popqa.py" > /dev/null; then
    echo "✅ 实验正在运行"
    echo ""
    
    # Show current process
    ps aux | grep run_popqa.py | grep -v grep | awk '{print "当前进程:", $11, $12, $13, $14, $15}'
    echo ""
else
    echo "❌ 实验未运行"
    echo ""
fi

# Show last 10 lines of log
echo "最新日志:"
echo "---"
tail -10 run_500.log
echo ""

# Check result files
echo "========================================="
echo "结果文件:"
echo "========================================="
ls -lh results/popqa_500/*.jsonl 2>/dev/null | awk '{print $9, "  ", $5}' || echo "暂无结果文件"
echo ""

# Count completed samples
if [ -f "results/popqa_500/baseline_500.jsonl" ]; then
    baseline_count=$(wc -l < results/popqa_500/baseline_500.jsonl)
    echo "Baseline: $baseline_count/500 样本"
fi

if [ -f "results/popqa_500/jes_500.jsonl" ]; then
    jes_count=$(wc -l < results/popqa_500/jes_500.jsonl)
    echo "JES: $jes_count/500 样本"
fi

if [ -f "results/popqa_500/force_adopt_500.jsonl" ]; then
    fa_count=$(wc -l < results/popqa_500/force_adopt_500.jsonl)
    echo "Force Adopt: $fa_count/500 样本"
fi

if [ -f "results/popqa_500/force_reject_500.jsonl" ]; then
    fr_count=$(wc -l < results/popqa_500/force_reject_500.jsonl)
    echo "Force Reject: $fr_count/500 样本"
fi

echo ""
echo "运行 'bash check_progress.sh' 查看最新进度"

