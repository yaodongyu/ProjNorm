import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import copy


def _weight_diff_norm_init(net, net_baseline):
    """
    Returns:
        the l2 norm difference the two networks
    """
    params1 = list(net.parameters())
    params2 = list(net_baseline.parameters())

    diff = 0
    for i in range(len(list(net.parameters()))):
        param1 = params1[i]
        param2 = params2[i]
        diff += (torch.norm(param1.flatten() - param2.flatten()) ** 2).cpu().detach().numpy()
    return np.sqrt(diff)


class ProjNorm(torch.nn.Module):
    """
    Projection Norm (ProjNorm)
    """
    def __init__(self, base_model):
        super(ProjNorm, self).__init__()
        self.base_model = copy.deepcopy(base_model)
        self.reference_model = copy.deepcopy(base_model)
        self.pseudo_model = None
        self.max_epochs = 1000

    def update_pseudo_model(self, data_loader, pseudo_model, lr, pseudo_iters):
        optimizer = optim.SGD(pseudo_model.parameters(),
                              lr=lr,
                              momentum=0.9,
                              weight_decay=0.0)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pseudo_iters)
        criterion = nn.CrossEntropyLoss().cuda()
        trainloader_iterator = iter(data_loader)

        for iteration in range(1, pseudo_iters + 1):
            pseudo_model.train()

            try:
                inputs, targets = next(trainloader_iterator)
            except StopIteration:
                trainloader_iterator = iter(data_loader)
                inputs, targets = next(trainloader_iterator)
            if iteration == 1:
                print('targets[:10]:', targets[:10])

            inputs = inputs.cuda()
            # pseudo-label by base_model
            _, pseudo_labels = self.base_model(inputs).max(1)
            pseudo_labels = pseudo_labels.detach()

            optimizer.zero_grad()
            outputs = pseudo_model(inputs)
            loss = criterion(outputs, pseudo_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss = loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total = pseudo_labels.size(0)
            correct = predicted.eq(pseudo_labels).sum().item()
            if iteration % 20 == 0:
                current_lr = 0.0
                for param_group in optimizer.param_groups:
                    current_lr = param_group['lr']
                print('iteration {}: train loss: {:.6f}, train acc: {:.6f}, current lr: {:.6f}'.format(iteration,
                                                                                                       train_loss / total,
                                                                                                       correct / total,
                                                                                                       current_lr))
        pseudo_model.eval()
        self.pseudo_model = copy.deepcopy(pseudo_model)
        print('========Pseudo-training finished========')

    def update_ref_model(self, data_loader, ref_model, lr, pseudo_iters):
        optimizer = optim.SGD(ref_model.parameters(),
                              lr=lr,
                              momentum=0.9,
                              weight_decay=0.0)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pseudo_iters)
        criterion = nn.CrossEntropyLoss().cuda()
        trainloader_iterator = iter(data_loader)

        for iteration in range(1, pseudo_iters + 1):
            ref_model.train()
            try:
                inputs, targets = next(trainloader_iterator)
            except StopIteration:
                trainloader_iterator = iter(data_loader)
                inputs, targets = next(trainloader_iterator)
            if iteration == 1:
                print('targets[:10]:', targets[:10])
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = ref_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss = loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total = targets.size(0)
            correct = predicted.eq(targets).sum().item()

            if iteration % 20 == 0:
                current_lr = 0.0
                for param_group in optimizer.param_groups:
                    current_lr = param_group['lr']
                print('iteration {}: train loss: {:.6f}, train acc: {:.6f}, current lr: {:.6f}'.format(iteration,
                                                                                                       train_loss / total,
                                                                                                       correct / total,
                                                                                                       current_lr))
        ref_model.eval()
        self.reference_model = copy.deepcopy(ref_model)
        print('========Pseudo-training (reference model) finished========')

    def compute_projnorm(self, model_ref, model_ood):
        return _weight_diff_norm_init(model_ref, model_ood)
