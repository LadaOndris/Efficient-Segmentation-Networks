#!/bin/bash
#PBS -N TrainSegmentationNetwork
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

echo "$PBS_JOBID is running on node $(hostname -f) in a scratch directory $EXECDIR" >> $FRONTEND/jobs_info.txt

echo "Running singularity..."
SINGULARITY_COMMANDS="pip3 install --cache-dir=$SCRATCHDIR -r requirements.txt && \
  export PYTHONPATH=$EXECDIR && \
  CUDA_VISIBLE_DEVICES=0 && \
  python3 train.py \
   --logger wandb \
   --max_epochs 300 \
   --dataset bdd100k \
   --num_workers $NUM_WORKERS \
   --train_type train"

singularity exec --nv -B "$SCRATCHDIR" /cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:22.12-py3.SIF  \
  bash -c "$SINGULARITY_COMMANDS"

echo "Copying $EXECDIR/checkpoint to $FRONTEND/..."
cp -r "$EXECDIR/checkpoint" "$FRONTEND/" || {
  echo >&2 "Couldn't copy log results to frontend."
  exit 3
}
echo "Cleaning scratch..."
clean_scratch
