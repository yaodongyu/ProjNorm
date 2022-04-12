import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import numpy as np


def load_cifar10_image(corruption_type,
                       clean_cifar_path,
                       corruption_cifar_path,
                       corruption_severity=0,
                       datatype='test',
                       num_samples=50000,
                       seed=1):
    """
    Returns:
        pytorch dataset object
    """
    assert datatype == 'test' or datatype == 'train'
    training_flag = True if datatype == 'train' else False

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    dataset = datasets.CIFAR10(clean_cifar_path,
                               train=training_flag,
                               transform=transform,
                               download=True)

    if corruption_severity > 0:
        assert not training_flag
        path_images = os.path.join(corruption_cifar_path, corruption_type + '.npy')
        path_labels = os.path.join(corruption_cifar_path, 'labels.npy')

        dataset.data = np.load(path_images)[(corruption_severity - 1) * 10000:corruption_severity * 10000]
        dataset.targets = list(np.load(path_labels)[(corruption_severity - 1) * 10000:corruption_severity * 10000])
        dataset.targets = [int(item) for item in dataset.targets]

    # randomly permute data
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    number_samples = dataset.data.shape[0]
    index_permute = torch.randperm(number_samples)
    dataset.data = dataset.data[index_permute]
    dataset.targets = np.array([int(item) for item in dataset.targets])
    dataset.targets = dataset.targets[index_permute].tolist()

    # randomly subsample data
    if datatype == 'train' and num_samples < 50000:
        indices = torch.randperm(50000)[:num_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
        print('number of training data: ', len(dataset))
    if datatype == 'test' and num_samples < 10000:
        indices = torch.randperm(10000)[:num_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
        print('number of test data: ', len(dataset))

    return dataset
