#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -p gpu
#SBATCH --gres=gpu:1


echo "Job started on $(hostname)"
echo "Time: $(date)"

# Load the cluster Python
module load python/3.9.9

module load cuda

# Define paths (keeps things tidy)
VENV_DIR=$HOME/arcn/venvs/benchmark_env
REPO_DIR=$HOME/arcn/GA_POCS

# Create venv ONLY if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python -m venv $VENV_DIR
fi

# Activate environment
source $VENV_DIR/bin/activate

# Install requirements ONLY if not already installed
# (checks for a marker file we create after first install)
if [ ! -f "$VENV_DIR/.requirements_installed" ]; then
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r $REPO_DIR/requirements.txt
    touch $VENV_DIR/.requirements_installed
else
    echo "Dependencies already installed. Skipping."
fi

# Run your code
cd $REPO_DIR/tests
python benchmark.py

echo "Finished at $(date)"
