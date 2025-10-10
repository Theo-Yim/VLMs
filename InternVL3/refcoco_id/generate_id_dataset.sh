#!/bin/bash
# Identity Tool-Calling Dataset Generation
# Parallelizes Stage 1 across multiple GPUs, then runs Stage 2

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
GPUS="0,1,2,3"  # GPUs to use (comma-separated)
MERGED_DATA="merged_refcoco_data.pkl"
COCO_PATH="/mnt/nas3/Data/coco"
OUTPUT_FOLDER="$COCO_PATH/refcoco_identity_stage1"
FINAL_OUTPUT="/workspace/VLMs/InternVL3/refcoco_id/dataset/identity_qa_pairs_31k.json"

# Calculate total images and splits
TOTAL_IMAGES=14274  # Total RefCOCO images (30956 total images but 14274 images have people)
NUM_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)
IMAGES_PER_GPU=$((TOTAL_IMAGES / NUM_GPUS))

echo "============================================================================"
echo "Identity Dataset Generation"
echo "============================================================================"
echo "Total RefCOCO images: $TOTAL_IMAGES"
echo "Images with people: ~14,274 (will be filtered by Stage 1)"
echo "GPUs: $GPUS"
echo "Num GPUs: $NUM_GPUS"
echo "Images per GPU: ~$IMAGES_PER_GPU"
echo "Output folder: $OUTPUT_FOLDER"
echo "============================================================================"

# ============================================================================
# Stage 1: Parallel Generation across GPUs
# ============================================================================
echo ""
echo "Starting Stage 1: Parallel Q&A generation across $NUM_GPUS GPUs..."
echo ""

# Create output folder
mkdir -p "$OUTPUT_FOLDER"

# Launch parallel processes
PIDS=()
GPU_ARRAY=($(echo $GPUS | tr ',' ' '))

for i in "${!GPU_ARRAY[@]}"; do
    GPU=${GPU_ARRAY[$i]}
    START=$((i * IMAGES_PER_GPU))

    # Last GPU gets remaining images
    if [ $i -eq $((NUM_GPUS - 1)) ]; then
        END=$TOTAL_IMAGES
    else
        END=$(((i + 1) * IMAGES_PER_GPU))
    fi

    echo "GPU $GPU: Processing images $START to $END"

    # Launch in background (filter vLLM progress bars)
    CUDA_VISIBLE_DEVICES=$GPU python /workspace/VLMs/InternVL3/refcoco_id/identity_stage1.py \
        --merged_data "$MERGED_DATA" \
        --coco_path "$COCO_PATH" \
        --output_folder "$OUTPUT_FOLDER" \
        --start $START \
        --end $END \
        --resume \
        2>&1 | grep -v -E "(Adding requests:|Processed prompts:)" > "/workspace/VLMs/InternVL3/refcoco_id/logs/stage1_gpu${GPU}.log" &

    PIDS+=($!)
done

echo ""
echo "Launched $NUM_GPUS parallel processes. Monitoring progress..."
echo "Logs: logs/stage1_gpu*.log"
echo ""

# Wait for all processes to complete
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    GPU=${GPU_ARRAY[$i]}
    echo "Waiting for GPU $GPU (PID: $PID)..."

    if wait $PID; then
        echo "✓ GPU $GPU completed successfully"
    else
        echo "✗ GPU $GPU failed (PID: $PID)"
        exit 1
    fi
done

echo ""
echo "============================================================================"
echo "Stage 1 Complete - All GPUs finished"
echo "============================================================================"

# ============================================================================
# Stage 2: Refinement (CPU only, fast)
# ============================================================================
echo ""
echo "Starting Stage 2: Refining and enriching answers..."
echo ""

python /workspace/VLMs/InternVL3/refcoco_id/identity_stage2.py \
    --stage1_folder "$OUTPUT_FOLDER" \
    --output "$FINAL_OUTPUT" \
    | tee /workspace/VLMs/InternVL3/refcoco_id/logs/stage2.log

echo ""
echo "============================================================================"
echo "Generation Complete!"
echo "============================================================================"
echo "Output: $FINAL_OUTPUT"
echo ""

# Show final statistics
if [ -f "$FINAL_OUTPUT" ]; then
    NUM_SAMPLES=$(python3 -c "import json; print(len(json.load(open('$FINAL_OUTPUT'))))")
    echo "Total Q&A samples generated: $NUM_SAMPLES"
fi

echo ""
echo "Logs:"
echo "  Stage 1: logs/stage1_gpu*.log"
echo "  Stage 2: logs/stage2.log"
echo "============================================================================"
