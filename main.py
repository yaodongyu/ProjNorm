import argparse

from projnorm import *
from load_data import *
from model import ResNet18
from utils import evaluation

"""# Configuration"""
parser = argparse.ArgumentParser(description='ProjNorm.')
parser.add_argument('--arch', default='resnet18', type=str)
parser.add_argument('--cifar_data_path', default='./data', type=str)
parser.add_argument('--cifar_corruption_path', default='./data/cifar/CIFAR-10-C', type=str)
parser.add_argument('--corruption', default='snow', type=str)
parser.add_argument('--severity', default=5, type=int)
parser.add_argument('--pseudo_iters', default=50, type=int)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--use_base_model', action='store_true',
                    default=False, help='apply base_model for computing ProjNorm')
args = vars(parser.parse_args())

if __name__ == "__main__":
    # setup valset_iid/val_ood loaders
    random_seeds = torch.randint(0, 10000, (2,))
    valset_iid = load_cifar10_image(corruption_type='clean',
                                    clean_cifar_path=args['cifar_data_path'],
                                    corruption_cifar_path=args['cifar_corruption_path'],
                                    corruption_severity=0,
                                    datatype='test',
                                    seed=random_seeds[0])
    val_iid_loader = torch.utils.data.DataLoader(valset_iid,
                                                 batch_size=args['batch_size'],
                                                 shuffle=True)

    valset_ood = load_cifar10_image(corruption_type=args['corruption'],
                                    clean_cifar_path=args['cifar_data_path'],
                                    corruption_cifar_path=args['cifar_corruption_path'],
                                    corruption_severity=args['severity'],
                                    datatype='test',
                                    seed=random_seeds[1])
    val_ood_loader = torch.utils.data.DataLoader(valset_ood,
                                                 batch_size=args['batch_size'],
                                                 shuffle=True)

    # init ProjNorm
    save_dir_path = './checkpoints/{}'.format(args['arch'])

    base_model = torch.load('{}/base_model.pt'.format(save_dir_path))
    base_model.eval()
    PN = ProjNorm(base_model=base_model)

    if not args['use_base_model']:
        ref_model = torch.load('{}/ref_model.pt'.format(save_dir_path))
        ref_model.eval()
        PN.reference_model = ref_model

    ################ train iid pseudo model ################
    if args['arch'] == 'resnet18':
        pseudo_model = ResNet18(num_classes=args['num_classes'], seed=args['seed']).cuda()
    else:
        raise ValueError('incorrect model name')

    PN.update_pseudo_model(val_iid_loader,
                           pseudo_model,
                           lr=args['lr'],
                           pseudo_iters=args['pseudo_iters'])

    # compute IID ProjNorm
    projnorm_value_iid = PN.compute_projnorm(PN.reference_model, PN.pseudo_model)

    ################ train ood pseudo model ################
    if args['arch'] == 'resnet18':
        pseudo_model = ResNet18(num_classes=args['num_classes'], seed=args['seed']).cuda()
    else:
        raise ValueError('incorrect model name')

    PN.update_pseudo_model(val_ood_loader,
                           pseudo_model,
                           lr=args['lr'],
                           pseudo_iters=args['pseudo_iters'])

    # compute OOD ProjNorm
    projnorm_value = PN.compute_projnorm(PN.reference_model, PN.pseudo_model)

    print('=============in-distribution=============')
    print('(in-distribution) ProjNorm value: ', projnorm_value_iid)
    test_loss_iid, test_error_iid = evaluation(net=base_model, testloader=val_iid_loader)
    print('(in-distribution) test error: ', test_error_iid)

    print('===========out-of-distribution===========')
    print('(out-of-distribution) ProjNorm value: ', projnorm_value)
    test_loss_ood, test_error_ood = evaluation(net=base_model, testloader=val_ood_loader)
    print('(out-of-distribution) test error: ', test_error_ood)
    print('========finished========')
