#!/bin/bash
# Monitor progress of identity dataset generation

OUTPUT_FOLDER="${1:-/mnt/nas3/Data/coco/refcoco_identity_stage1}"
TOTAL_IMAGES=14274  # Num of images with people out of 30956 Total RefCOCO images

echo "Monitoring: $OUTPUT_FOLDER"
echo ""

while true; do
    # Count completed files
    COMPLETED=$(find "$OUTPUT_FOLDER" -name "*.json" 2>/dev/null | wc -l)
    PROGRESS=$((COMPLETED * 100 / TOTAL_IMAGES))

    # Clear screen and show progress
    clear
    echo "============================================================================"
    echo "Identity Dataset Generation - Progress Monitor"
    echo "============================================================================"
    echo ""
    echo "Completed: $COMPLETED / $TOTAL_IMAGES images ($PROGRESS%)"
    echo ""

    # Show per-GPU progress from logs
    if [ -d "logs" ]; then
        echo "Per-GPU Status:"
        echo "----------------------------------------"
        for log in logs/stage1_gpu*.log; do
            if [ -f "$log" ]; then
                GPU=$(basename "$log" | sed 's/stage1_gpu\(.*\)\.log/\1/')
                LAST_LINE=$(tail -1 "$log" 2>/dev/null)
                echo "GPU $GPU: $LAST_LINE"
            fi
        done
        echo ""
    fi

    # Show recent errors
    if [ -d "logs" ]; then
        ERRORS=$(grep -i "error\|exception\|failed" logs/stage1_gpu*.log 2>/dev/null | tail -3)
        if [ -n "$ERRORS" ]; then
            echo "Recent errors:"
            echo "----------------------------------------"
            echo "$ERRORS"
            echo ""
        fi
    fi

    # Estimate time remaining (rough estimate: 30 sec per image)
    REMAINING=$((TOTAL_IMAGES - COMPLETED))
    TIME_SEC=$((REMAINING * 30))
    TIME_HOURS=$((TIME_SEC / 3600))
    echo "Estimated time remaining: ~$TIME_HOURS hours"
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    echo "============================================================================"

    # Break if complete
    if [ $COMPLETED -eq $TOTAL_IMAGES ]; then
        echo ""
        echo "✓ Generation complete!"
        break
    fi

    sleep 10
done
