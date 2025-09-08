#!/bin/sh

# Ultra-Simple Parallel Launcher
echo "=== Simple Parallel InternVL Launcher ==="

# EDIT THESE PATHS FOR YOUR SETUP
DATA_ROOT="/home/Theo-Yim/data/lh-poc/"
# IMAGE_ROOT="/mnt/nas1/data/lh-poc/lh-data-image/image/20250722"
RESULT_DIR="/workspace/VLMs/utils/lh-poc/results_theo_parallel"
MODEL_PATH="OpenGVLab/InternVL3_5-38B"
PYTHON_SCRIPT="inference_theo_pa.py"

export PYTHONPATH=/workspace/VLMs/:$PYTHONPATH

# MANUALLY SET WHICH GPUs TO USE (space-separated list)
# Leave empty to use all GPUs: USE_GPUS=""
# Or specify GPUs: USE_GPUS="0 1 3 5"
USE_GPUS="0 1 2 3 4 5 6 7"

# Check script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: $PYTHON_SCRIPT not found!"
    exit 1
fi

# Set up GPU list
if [ -z "$USE_GPUS" ]; then
    # Auto-detect all GPUs
    echo "Auto-detecting GPUs..."
    TOTAL_GPUS=`nvidia-smi --list-gpus | wc -l`
    echo "Found $TOTAL_GPUS GPUs - using all"
    
    # Create list "0 1 2 3..."
    GPU_LIST=""
    i=0
    while [ $i -lt $TOTAL_GPUS ]; do
        if [ -z "$GPU_LIST" ]; then
            GPU_LIST="$i"
        else
            GPU_LIST="$GPU_LIST $i"
        fi
        i=`expr $i + 1`
    done
else
    # Use manually specified GPUs
    GPU_LIST="$USE_GPUS"
    echo "Using manually specified GPUs: $GPU_LIST"
fi

# Count how many GPUs we're using
GPU_COUNT=0
for gpu in $GPU_LIST; do
    GPU_COUNT=`expr $GPU_COUNT + 1`
done

echo "Will use $GPU_COUNT GPUs: $GPU_LIST"

# Create directories
mkdir -p "$RESULT_DIR"
mkdir -p "$RESULT_DIR/logs" 

# Get dataset size
echo "Counting dataset..."
cat > /tmp/count_data.py << 'EOF'
import sys
import io
import contextlib
sys.path.append('.')

# Suppress all output except our final number
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    try:
        from dataloader import LHDataLoader
        loader = LHDataLoader(sys.argv[1], sys.argv[2])
        count = len(loader)
    except Exception as e:
        count = 0

# Only print the number
print(count)
EOF

TOTAL_ITEMS=`python3 /tmp/count_data.py "$DATA_ROOT" "train" 2>/dev/null | tail -1`
rm /tmp/count_data.py

echo "Total items: $TOTAL_ITEMS"

if [ $TOTAL_ITEMS -eq 0 ]; then
    echo "ERROR: No data found!"
    exit 1
fi

# Calculate split
ITEMS_PER_GPU=`expr $TOTAL_ITEMS / $GPU_COUNT`
echo "Items per GPU: $ITEMS_PER_GPU"

# Launch processes
echo "Starting processes..."

process_id=0
start=0

for gpu_id in $GPU_LIST; do
    end=`expr $start + $ITEMS_PER_GPU`
    
    # Last process gets remainder
    if [ $process_id -eq `expr $GPU_COUNT - 1` ]; then
        end=$TOTAL_ITEMS
    fi
    
    echo "Process $process_id on GPU $gpu_id: items $start to `expr $end - 1`"
    
    # Start process
    python3 "$PYTHON_SCRIPT" \
        --data_root "$DATA_ROOT" \
        --result_dir "$RESULT_DIR" \
        --model_path "$MODEL_PATH" \
        --gpu_id $gpu_id \
        --start_idx $start \
        --end_idx $end \
        --enable_thinking \
        --process_id $process_id > "$RESULT_DIR/logs/process_$process_id.log" 2>&1 &
    
    echo "Started process $process_id on GPU $gpu_id (log: process_$process_id.log)"
    
    start=$end
    process_id=`expr $process_id + 1`
    sleep 1
done

echo ""
echo "All $GPU_COUNT processes started!"
echo "Monitor: tail -f $RESULT_DIR/logs/process_*.log"
echo "Waiting for completion..."

# Wait for all
wait

echo ""
echo "Done! Results in: $RESULT_DIR"