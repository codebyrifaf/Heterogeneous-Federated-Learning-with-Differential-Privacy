from utils.plot_utils import *
import argparse

from data.Mnist.data_generator import generate_data as generate_mnist_data
from data.Mnist.data_generator import generate_pca_data as generate_mnist_pca_data
from data.CIFAR_10.data_generator import generate_data as generate_cifar10_data