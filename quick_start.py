# Quick Start Script for Differential Privacy Federated Learning
# This script runs a minimal example to test the complete pipeline

import sys
import os

# Ensure we're in the right directory
os.chdir(r'd:\Differential-Privacy-for-Heterogeneous-Federated-Learning-main\Differential-Privacy-for-Heterogeneous-Federated-Learning-main')

print("=" * 80)
print("QUICK TEST RUN - Differential Privacy Federated Learning")
print("=" * 80)

# ===== STEP 1: Generate Small Dataset =====
print("\n[STEP 1/3] Generating small synthetic dataset...")
print("-" * 80)

from data.Logistic.data_generator import generate_data

generate_data(
    num_users=10,
    same_sample_size=True,
    num_samples=500,
    dim_input=40,
    dim_output=10,
    alpha=0.0,
    beta=0.0,
    number=0,  # Dataset ID
    normalise=True,
    standardize=True,
    iid=False
)

print("\n[SUCCESS] Data generated successfully!")
print("Location: data/Logistic/data/train/ and data/Logistic/data/test/")

# ===== STEP 2: Train Centralized Baseline =====
print("\n[STEP 2/3] Training centralized baseline model...")
print("-" * 80)

from simulate import find_optimum

find_optimum(
    dataset="Logistic",
    model="mclr",
    number=0,
    dim_input=40,
    dim_output=10,
    alpha=0.0,
    beta=0.0
)

print("\n[SUCCESS] Baseline model trained!")
print("Location: models/Logistic/mclr/server_lowest_(0.0, 0.0).pt")

# ===== STEP 3: Run Quick Federated Learning Experiment =====
print("\n[STEP 3/3] Running federated learning experiment...")
print("Algorithm: FedAvg (without DP)")
print("Communication rounds: 20")
print("Local updates: 5")
print("-" * 80)

from simulate import simulate

simulate(
    dataset="Logistic",
    algorithm="FedAvg",
    model="mclr",
    dim_input=40,
    dim_output=10,
    nb_users=10,
    nb_samples=500,
    sample_ratio=0.2,
    user_ratio=0.2,
    weight_decay=5e-3,
    local_learning_rate=1.0,
    max_norm=1.0,
    local_updates=5,
    noise=False,
    times=1,
    dp="None",
    sigma_gaussian=50.0,
    dim_pca=None,
    similarity=None,
    alpha=0.0,
    beta=0.0,
    number=0,
    num_glob_iters=20,
    time=0
)

print("\n" + "=" * 80)
print("[ALL DONE!] Quick test completed successfully!")
print("=" * 80)

