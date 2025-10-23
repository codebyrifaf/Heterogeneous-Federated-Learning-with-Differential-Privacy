"""
Quick Training Script with Real-time Visualization
Run this to see fancy graphs while training!
"""

import torch
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from flearn.servers.optimum_with_viz import OptimWithVisualization
from flearn.trainmodel.models import MclrLogistic

print("=" * 80)
print("🎨 TRAINING WITH REAL-TIME VISUALIZATION")
print("=" * 80)
print("\n📋 Instructions:")
print("1. This script will start training")
print("2. Open ANOTHER terminal and run: tensorboard --logdir=runs")
print("3. Open browser to: http://localhost:6006")
print("4. Watch the fancy graphs update in real-time!")
print("=" * 80)

input("\nPress ENTER to start training...")

# Configuration
dataset = "Logistic"
model_name = "mclr"
dim_input = 40
dim_output = 10
alpha = 0.0
beta = 0.0
number = 0

print(f"\n🚀 Starting training for {dataset} dataset...")

# Create model
model = MclrLogistic(input_dim=dim_input, output_dim=dim_output), model_name

# Create trainer with visualization
use_cuda = torch.cuda.is_available()
if use_cuda:
    print("✅ Using GPU")
else:
    print("💻 Using CPU")

trainer = OptimWithVisualization(
    dataset=dataset,
    model=model,
    number=number,
    similarity=None,
    alpha=alpha,
    beta=beta,
    dim_pca=None,
    use_cuda=use_cuda
)

# Train
trainer.train()

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE!")
print("=" * 80)
print("\n📊 To view results:")
print("   1. If TensorBoard is not running, open another terminal")
print("   2. Run: tensorboard --logdir=runs")
print("   3. Open: http://localhost:6006")
print("\n🎯 You'll see graphs for:")
print("   - Training Loss (per batch and per epoch)")
print("   - Test Loss")
print("   - Test Accuracy")
print("   - Best Training Loss")
print("=" * 80)
