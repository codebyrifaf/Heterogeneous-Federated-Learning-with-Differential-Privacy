from utils.plot_utils import *
import argparse
from simulate import simulate
from simulate import simulate_cross_validation
from simulate import find_optimum
from data.Mnist.data_generator import generate_data as generate_mnist_data
from data.Mnist.data_generator import generate_pca_data as generate_mnist_pca_data
from data.Femnist.data_generator import generate_data as generate_femnist_data
from data.Femnist.data_generator import generate_pca_data as generate_femnist_pca_data
from data.CIFAR_10.data_generator import generate_data as generate_cifar10_data
from data.Logistic.data_generator import generate_data as generate_logistic_data


def generate_data(dataset, nb_users, nb_samples, dim_input=40, dim_output=10, similarity=1.0, alpha=0., beta=0.,
                  number=0, iid=False, same_sample_size=True, normalise=False, standardize=False):
    if dataset == 'Femnist':
        generate_femnist_data(similarity, num_users=nb_users, num_samples=nb_samples, number=number,
                              normalise=normalise)
    elif dataset == 'Mnist':
        generate_mnist_data(similarity, num_users=nb_users, num_samples=nb_samples, number=number,
                            normalise=normalise)
    elif dataset == 'CIFAR_10':
        generate_cifar10_data(similarity, num_users=nb_users, num_samples=nb_samples, number=number)

    elif dataset == 'Logistic':
        generate_logistic_data(num_users=nb_users, same_sample_size=same_sample_size, num_samples=nb_samples,
                               dim_input=dim_input, dim_output=dim_output, alpha=alpha, beta=beta, number=number,
                               normalise=normalise, standardize=standardize, iid=iid)


def run_simulation(time, dataset, algo, model, similarity, alpha, beta, number, dim_input, dim_output, same_sample_size,
                   nb_users, user_ratio, nb_samples, sample_ratio, local_updates, weight_decay, local_learning_rate,
                   max_norm, dp, sigma_gaussian, normalise, standardize, times, optimum, num_glob_iters, generate,
                   generate_pca, dim_pca, tuning, learning, plot):
    if dataset == "Femnist":
        nb_users = 40
        nb_samples = 2500

    if dataset == "Mnist":
        nb_users = 60
        nb_samples = 1000

    if dataset == "CIFAR_10":
        nb_users = 50
        nb_samples = 1000

    # FEMNIST DATA
    # Potential models : mclr, NN1, NN1_PCA

    femnist_dict = {"dataset": "Femnist",
                    "model": model,
                    "dim_input": 784,
                    "dim_pca": dim_pca,
                    "dim_output": 47,
                    "nb_users": nb_users,
                    "nb_samples": nb_samples,
                    "sample_ratio": sample_ratio,
                    "local_updates": local_updates,
                    "user_ratio": user_ratio,
                    "weight_decay": weight_decay,
                    "local_learning_rate": local_learning_rate,
                    "max_norm": max_norm}

    # MNIST DATA
    # Potential models : mclr, NN1, NN1_PCA

    mnist_dict = {"dataset": "Mnist",
                  "model": model,
                  "dim_input": 784,
                  "dim_pca": dim_pca,
                  "dim_output": 10,
                  "nb_users": nb_users,
                  "nb_samples": nb_samples,
                  "sample_ratio": sample_ratio,
                  "local_updates": local_updates,
                  "user_ratio": user_ratio,
                  "weight_decay": weight_decay,
                  "local_learning_rate": local_learning_rate,
                  "max_norm": max_norm}

    # CIFAR-10 DATA
    # Potential models : CNN

    cifar10_dict = {"dataset": "CIFAR_10",
                    "model": "CNN",
                    "dim_input": 1024,
                    "dim_pca": None,
                    "dim_output": 10,
                    "nb_users": nb_users,
                    "nb_samples": nb_samples,
                    "sample_ratio": sample_ratio,
                    "local_updates": local_updates,
                    "user_ratio": user_ratio,
                    "weight_decay": weight_decay,
                    "local_learning_rate": local_learning_rate,
                    "max_norm": max_norm}

    # SYNTHETIC DATA
    # only one model : mclr

    logistic_dict = {"dataset": "Logistic",
                     "model": "mclr",
                     "dim_input": dim_input,
                     "dim_pca": None,
                     "dim_output": dim_output,
                     "nb_users": nb_users,
                     "nb_samples": nb_samples,
                     "sample_ratio": sample_ratio,
                     "local_updates": local_updates,
                     "user_ratio": user_ratio,
                     "weight_decay": weight_decay,
                     "local_learning_rate": local_learning_rate,
                     "max_norm": max_norm}