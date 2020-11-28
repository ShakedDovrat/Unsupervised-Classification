import argparse
import os

import torch
import torch.nn as nn
from termcolor import colored

from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset,\
                                get_val_dataset, get_train_dataloader,\
                                get_val_dataloader, get_train_transformations,\
                                get_val_transformations, get_optimizer,\
                                adjust_learning_rate
from utils.train_utils import simclr_fine_tune_train
from utils.utils import AverageMeter, confusion_matrix


class AttributesHead(nn.Module):
    def __init__(self, features_dim, num_classes):
        super(AttributesHead, self).__init__()
        self.fc = nn.Linear(features_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


@torch.no_grad()
def attributes_evaluate(val_loader, model):
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()

    for batch in val_loader:
        images = batch['image'].cuda(non_blocking=True)
        target = torch.from_numpy(batch['target']).cuda(non_blocking=True)#batch['target'].cuda(non_blocking=True)

        output = model(images)
        # output = memory_bank.weighted_knn(output)#.cpu())

        acc1 = 100*torch.mean(torch.eq(torch.argmax(output, dim=1), target).float())
        top1.update(acc1.item(), images.size(0))

    return top1.avg


def main(args):
    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))

    # Model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print(model)
    # from torchsummary import summary
    # summary(model, (3, p['transformation_kwargs']['crop_size'], p['transformation_kwargs']['crop_size']))
    model = model.cuda()

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations(p)
    print('Train transforms:', train_transforms)
    val_transforms = get_val_transformations(p)
    print('Validation transforms:', val_transforms)
    train_dataset = get_train_dataset(p, train_transforms, to_augmented_dataset=True,
                                      split='train+unlabeled')  # Split is for stl-10
    val_dataset = get_val_dataset(p, val_transforms)
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))

    # Criterion
    print(colored('Retrieve criterion', 'blue'))
    criterion = get_criterion(p)
    print('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.cuda()

    # Optimizer and scheduler
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # Checkpoint
    assert os.path.exists(p['pretext_checkpoint']), "Checkpoint not found - can't fine-tune."
    print(colored('Restart from checkpoint {}'.format(p['pretext_checkpoint']), 'blue'))
    checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.load_state_dict(checkpoint['model'])
    model.cuda()
    # start_epoch = checkpoint['epoch']
    start_epoch = 0

    # Train linear model from representations to evaluate attributes classification
    print(colored('Train linear', 'blue'))

    for parameter in model.parameters():
        parameter.requires_grad = False
    model = nn.Sequential(model, AttributesHead(p['model_kwargs']['features_dim'], p['num_attribute_classes']))
    model.cuda()

    # Training
    print(colored('Starting main loop', 'blue'))
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' % (epoch, p['epochs']), 'yellow'))
        print(colored('-' * 15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        print('Train ...')
        simclr_fine_tune_train(train_dataloader, model, criterion, optimizer, epoch)

        # Evaluate
        acc = attributes_evaluate(val_dataloader, model)
        print('Val set accuracy %.2f' % acc)

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1}, p['pretext_fine_tune_checkpoint'])

    # Save final model
    torch.save(model.state_dict(), p['pretext_fine_tune_model'])


if __name__ == '__main__':
    # Parser
    parser = argparse.ArgumentParser(description='SimCLR')
    parser.add_argument('--config_env',
                        help='Config file for the environment')
    parser.add_argument('--config_exp',
                        help='Config file for the experiment')
    args = parser.parse_args()

    main(args)
