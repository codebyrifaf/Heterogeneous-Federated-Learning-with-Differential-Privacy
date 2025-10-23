"""
Training with Live Matplotlib Graphs (No Browser Needed!)
This shows graphs directly in a window while training
"""

import torch
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))

from flearn.servers.optimum import Optim
from flearn.trainmodel.models import MclrLogistic

class LivePlotter:
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.test_accuracies = []
        self.epochs = []
        
        # Set up the plot
        plt.ion()  # Interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle('🎯 Real-Time Training Visualization', fontsize=16, fontweight='bold')
        
        # Loss plot
        self.line1, = self.ax1.plot([], [], 'b-', label='Train Loss', linewidth=2)
        self.line2, = self.ax1.plot([], [], 'r-', label='Test Loss', linewidth=2)
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('📉 Training & Test Loss')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        self.line3, = self.ax2.plot([], [], 'g-', label='Test Accuracy', linewidth=2)
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy (%)')
        self.ax2.set_title('📈 Test Accuracy')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        
    def update(self, epoch, train_loss, test_loss, test_accuracy):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.test_accuracies.append(test_accuracy)
        
        # Update loss plot
        self.line1.set_data(self.epochs, self.train_losses)
        self.line2.set_data(self.epochs, self.test_losses)
        self.ax1.relim()
        self.ax1.autoscale_view()
        
        # Update accuracy plot
        self.line3.set_data(self.epochs, self.test_accuracies)
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # Add current values as text
        self.ax1.set_title(f'📉 Training & Test Loss | Current: Train={train_loss:.4f}, Test={test_loss:.4f}')
        self.ax2.set_title(f'📈 Test Accuracy | Current: {test_accuracy:.2f}%')
        
        plt.pause(0.01)
        
    def save(self, filename='training_results.png'):
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n💾 Graph saved to: {filename}")

# Global plotter
plotter = None

# Monkey-patch the Optim class to add visualization
original_train = Optim.train

def train_with_viz(self):
    global plotter
    plotter = LivePlotter()
    
    import numpy as np
    import copy
    
    lowest_loss = np.inf
    
    for epoch in range(int(self.epochs)):
        self.model.train()
        average_loss = 0
        count = 0

        # Train
        for i, (images, labels) in enumerate(self.train_loader):
            if self.use_cuda:
                images, labels = images.cuda(), labels.cuda()

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()

            if i % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(images), len(self.train_loader.dataset),
                           100. * i / len(self.train_loader), loss.data.item()))
            average_loss += loss.data.item()
            count += 1

        average_loss = average_loss / count

        if average_loss < lowest_loss:
            self.save_model()
            lowest_loss = copy.copy(average_loss)

        # Test
        correct = 0
        total = 0
        all_loss = 0
        count = 0

        self.model.eval()

        for i, (images, labels) in enumerate(self.test_loader_full):
            if self.use_cuda:
                images, labels = images.cuda(), labels.cuda()
            outputs = self.model(images)
            loss = self.loss(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            all_loss += loss.item()
            count += 1
            
        accuracy = 100 * correct / total
        all_loss = all_loss / count

        print('TEST : Epoch: {}. Loss: {}. Accuracy: {}.'.format(epoch, all_loss, accuracy))
        
        # Update visualization
        plotter.update(epoch, average_loss, all_loss, accuracy.item() if torch.is_tensor(accuracy) else accuracy)
    
    # Save final graph
    plotter.save('training_results.png')
    print("\n✅ Training complete! Graph window will stay open. Close it to exit.")
    plt.ioff()
    plt.show()

# Apply the patch
Optim.train = train_with_viz

if __name__ == "__main__":
    print("=" * 80)
    print("🎨 LIVE TRAINING VISUALIZATION (Matplotlib)")
    print("=" * 80)
    print("\n📊 A window will open showing real-time graphs!")
    print("   - Training Loss & Test Loss (top graph)")
    print("   - Test Accuracy (bottom graph)")
    print("\n⚠️  Keep the graph window open during training")
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
    
    print(f"\n🚀 Starting training...")
    
    # Create model
    model = MclrLogistic(input_dim=dim_input, output_dim=dim_output), model_name
    
    # Create trainer
    use_cuda = torch.cuda.is_available()
    trainer = Optim(
        dataset=dataset,
        model=model,
        number=number,
        similarity=None,
        alpha=alpha,
        beta=beta,
        dim_pca=None,
        use_cuda=use_cuda
    )
    
    # Train with visualization
    trainer.train()
