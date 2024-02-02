#!/bin/bash
#PBS -N PredictSegScoresOnBDD100k
#PBS -q gpu
#PBS -l select=1:ncpus=12:ngpus=1:mem=32gb:scratch_local=40gb:cl_adan=True
#PBS -l walltime=24:00:00
#PBS -m abe

NUM_WORKERS=11

FRONTEND=$PBS_O_WORKDIR
SOURCES=$FRONTEND/Efficient-Segmentation-Networks
EXECDIR=$SCRATCHDIR/Efficient-Segmentation-Networks

echo "Frontend: $FRONTEND"
echo "Sources: $SOURCES"
echo "Scratchdir: $SCRATCHDIR"

echo "Copying sources $SOURCES to $SCRATCHDIR..."
cp -r $SOURCES $SCRATCHDIR
cd "$EXECDIR"

echo "[PredictSegScoresOnBDD100k] $PBS_JOBID is running on node $(hostname -f) in a scratch directory $EXECDIR" >> $FRONTEND/jobs_info.txt

echo "Running singularity..."
SINGULARITY_COMMANDS="pip3 install --cache-dir=$SCRATCHDIR -r requirements.txt && \
  export PYTHONPATH=$EXECDIR && \
  CUDA_VISIBLE_DEVICES=0 && \
   python3 predict_scores_on_bdd100k.py \
    --model ERFNet \
    --dataset bdd100k \
    --num_workers $NUM_WORKERS \
    --batch_size 8 \
    --checkpoint checkpoint/ERFNet/19947564/ERFNetbs8gpu1_train/model_3.pth \
    --save_file_name iou_results_erfnet_3.csv"

singularity exec --nv -B "$SCRATCHDIR" /cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:22.12-py3.SIF  \
  bash -c "$SINGULARITY_COMMANDS"

echo "Copying $EXECDIR/result to $FRONTEND/"
cp -r "$EXECDIR/result" "$FRONTEND/" || {
  echo >&2 "Couldn't copy log results to frontend."
  exit 3
}
echo "Cleaning scratch..."
clean_scratch
