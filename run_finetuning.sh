#!/bin/bash

# ====================================================================================
# JTwBio - Universal Experiment Launcher
#
# This script launches multiple torchrun fine-tuning jobs in parallel.
# It assigns a unique port for each (model, experiment) combination and uses the
# correct Hydra configuration.
#
# Usage:
#   ./run_experiments.sh -m "model1 model2" -e "exp1 exp2"
#
# Example:
#   # Run simclr and chronos on uci_bp and wesad_hr
#   ./run_experiments.sh -m "simclr chronos" -e "uci_bp wesad_hr"
#
#   # Run all model configurations on all experiment configurations
#   ./run_experiments.sh
# ====================================================================================

BASE_PORT=29500
NPROC_PER_NODE=2

# Map model names to their corresponding training script modules
declare -A MODEL_TO_SCRIPT
MODEL_TO_SCRIPT=(
    [simclr]="baselines.SimCLR.run_simclr_finetuning"
    [byol]="baselines.BYOL.run_byol_finetuning"
    [simsiam]="baselines.SimSiam.run_simsiam_finetuning"
    [tfc]="baselines.TFC.run_tfc_finetuning"
    [encoder]="src.modules.encoder_finetuning"
    [mocov3]="baselines.MoCo-v3.run_mocov3_finetuning"
    [heartlang]="baselines.HeartLang.run_heartlang_finetuning"
    [papagei]="baselines.PaPaGei.run_papagei_finetuning"
)

# --- Script Logic ---

# Parse command-line options
while getopts "m:e:h" opt; do
  case ${opt} in
    m )
      MODELS_TO_RUN="$OPTARG"
      ;;
    e )
      EXPERIMENTS_TO_RUN="$OPTARG"
      ;;
    h )
      echo "Usage: $0 [-m \"model1 model2\"] [-e \"exp1 exp2\"]"
      exit 0
      ;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      exit 1
      ;;
  esac
done

# # Auto-discover models if not specified via -m
# if [ -z "$MODELS_TO_RUN" ]; then
#     echo "No models specified with -m. Auto-discovering from 'conf/model/' directory..."
#     # ls: list files
#     # grep -v: exclude config.yaml
#     # sed: remove .yaml extension
#     # tr: convert newlines to spaces
#     MODELS_TO_RUN=$(ls conf/model | grep -v 'config.yaml' | sed 's/\.yaml//' | tr '\n' ' ')
# fi

# # Auto-discover experiments if not specified via -e
# if [ -z "$EXPERIMENTS_TO_RUN" ]; then
#     echo "No experiments specified with -e. Auto-discovering from 'conf/experiment/' directory..."
#     EXPERIMENTS_TO_RUN=$(ls conf/experiment | sed 's/\.yaml//' | tr '\n' ' ')
# fi

echo "================================================="
echo "Models to run: $MODELS_TO_RUN"
echo "Experiments to run: $EXPERIMENTS_TO_RUN"
echo "================================================="
echo ""

pids=()
job_counter=0

for model in $MODELS_TO_RUN; do
    for experiment in $EXPERIMENTS_TO_RUN; do
        
        # Check if a script is defined for the model
        if [ -z "${MODEL_TO_SCRIPT[$model]}" ]; then
            echo "ERROR: No training script found for model '$model'. Please add it to the MODEL_TO_SCRIPT map."
            continue
        fi

        # Calculate a unique port for the job
        port=$((BASE_PORT + job_counter + 10 ))
        
        # Get the script module path for the current model
        script_module=${MODEL_TO_SCRIPT[$model]}
        
        echo "--> [Job ${job_counter}] Launching: model=${model}, experiment=${experiment} @ localhost:${port}"

        # Launch the torchrun command in the background (&)
        CUDA_VISIBLE_DEVICES=2,3 \
        torchrun --nproc_per_node=${NPROC_PER_NODE} \
                 --rdzv_endpoint="localhost:${port}" \
                 -m ${script_module} \
                 hydra.job.chdir=True \
                 model=${model} \
                 experiment=${experiment} &

        # Store the Process ID (PID) of the background job
        pids+=($!)
        
        # Increment the job counter
        job_counter=$((job_counter + 1))

        # Brief pause to prevent all jobs from starting simultaneously and scrambling the output
        sleep 2
    done
done

echo ""
echo "All ${job_counter} jobs have been launched in the background. Waiting for them to complete..."
echo "Process PIDs: ${pids[*]}"

# Wait for all background processes to finish
for pid in "${pids[@]}"; do
    wait $pid
done

echo ""
echo "================================================="
echo "All experiments have completed!"
echo "================================================="
