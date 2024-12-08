#!/bin/bash
#SBATCH -A cis240102p           # Project ID
#SBATCH -p GPU-shared           # Select GPU-shared partition
#SBATCH -N 1                    # Request 1 node
#SBATCH -n 5                    # Request 5 cores
#SBATCH --gres=gpu:4            # Request 4 GPUs
#SBATCH -t 00:10:00             # Set maximum run time to 10 minutes
#SBATCH --mail-user=milburnj@udel.edu  # Replace with your email address
#SBATCH --mail-type=ALL         # Notification type for emails

# Load CUDA module
module load cuda/12.4.0           # Load the appropriate CUDA module (adjust version as needed)

# Compile the matrix multiplication program
echo "Compiling GPU Matrix Multiplication Code"
nvcc -o matrix_multiply cublas.cpp -lcublas -lcudart

# Check if compilation succeeded
if [ $? -ne 0 ]; then
  echo "Compilation failed. Exiting."
  exit 1
fi

# Run the matrix multiplication program with example input sizes
echo "Running matrix multiplication with matrix size 1000x1000"
./matrix_multiply 1000

