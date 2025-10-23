import os
import torch
import torch.nn as nn
import numpy as np
import copy
import random
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from utils.model_utils import read_data, read_user_data
from torch.utils.tensorboard import SummaryWriter


class OptimWithVisualization:
    def __init__(self, dataset, model, number, similarity, alpha, beta, dim_pca, use_cuda):

        if similarity is None:
            similarity = (alpha, beta)
        if alpha < 0 and beta < 0:
            similarity = "iid"

        self.similarity = similarity
        self.dataset = dataset
        self.number = str(number)
        self.model = copy.deepcopy(model[0])
        if use_cuda:
            self.model = self.model.cuda()
        self.learning_rate = 0.02
        self.model_name = model[1]
        self.use_cuda = use_cuda

        # TensorBoard writer for visualization
        log_dir = f'runs/{dataset}_{model[1]}_baseline_{similarity}'
        self.writer = SummaryWriter(log_dir)
        print(f"\n🎨 TensorBoard logging enabled!")
        print(f"📊 To view real-time graphs, open another terminal and run:")
        print(f"   tensorboard --logdir=runs")
        print(f"   Then open: http://localhost:6006\n")

        # using PCA or not
        if model[1][-3:] != "PCA":
            dim_pca = None

        if model[1] == 'mclr':
            self.loss = nn.NLLLoss()
            self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            self.loss = nn.CrossEntropyLoss()
            self.optimizer = Adam(self.model.parameters())

        # Initialize data
        data = read_data(dataset, self.number, str(self.similarity), dim_pca)
        total_users = len(data[0])
        self.train_dataset = []
        self.test_dataset = []

        for i in range(total_users):
            id, train, test = read_user_data(i, data, dataset)
            self.train_dataset += train
            self.test_dataset += test

        random.shuffle(self.train_dataset)
        random.shuffle(self.test_dataset)

        self.batch_size = 500
        self.epochs = 50  # Reduced for faster training

        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.train_loader_full = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size)
        self.test_loader_full = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size)
        self.log_interval = round(len(self.train_loader.dataset) / (6 * self.batch_size))

    def train(self):
        lowest_loss = np.inf
        best_accuracy = 0
        
        for epoch in range(int(self.epochs)):
            self.model.train()

            average_loss = 0
            count = 0
            batch_losses = []

            # train
            for i, (images, labels) in enumerate(self.train_loader):
                if self.use_cuda:
                    images, labels = images.cuda(), labels.cuda()

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss(outputs, labels)
                loss.backward()
                self.optimizer.step()

                batch_loss = loss.data.item()
                batch_losses.append(batch_loss)
                
                # Log to TensorBoard every batch
                global_step = epoch * len(self.train_loader) + i
                self.writer.add_scalar('Loss/train_batch', batch_loss, global_step)

                if i % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, i * len(images), len(self.train_loader.dataset),
                               100. * i / len(self.train_loader), batch_loss))
                average_loss += batch_loss
                count += 1

            average_loss = average_loss / count

            # Log epoch metrics to TensorBoard
            self.writer.add_scalar('Loss/train_epoch', average_loss, epoch)

            if average_loss < lowest_loss:
                self.save_model()
                lowest_loss = copy.copy(average_loss)

            # test
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

            # Log test metrics to TensorBoard
            self.writer.add_scalar('Loss/test', all_loss, epoch)
            self.writer.add_scalar('Accuracy/test', accuracy, epoch)
            self.writer.add_scalar('Loss/best_train_loss', lowest_loss, epoch)

            # Track best accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy

            print('TEST : Epoch: {}. Loss: {}. Accuracy: {}. Best Acc: {:.2f}'.format(
                epoch, all_loss, accuracy, best_accuracy))

        # Close TensorBoard writer
        self.writer.close()
        print(f"\n✅ Training complete! Final best accuracy: {best_accuracy:.2f}%")
        print(f"📊 View results in TensorBoard at: http://localhost:6006")

    def save_model(self):
        model_path = os.path.join("models", self.dataset, self.model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server_lowest_" + str(self.similarity) + ".pt"))
