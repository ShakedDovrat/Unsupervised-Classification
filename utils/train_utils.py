"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import numpy as np
from utils.utils import AverageMeter, ProgressMeter, np_to_tensor_safe


def simclr_train(train_loader, model, criterion, optimizer, epoch, augs_criterion=None):
    """ 
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')
    if augs_criterion is None:
        progress = ProgressMeter(len(train_loader),
                                 [losses],
                                 prefix="Epoch: [{}]".format(epoch))
    else:
        augs_losses = AverageMeter('Augs Loss', ':.4e')
        progress = ProgressMeter(len(train_loader),
                                 [losses, augs_losses],
                                 prefix="Epoch: [{}]".format(epoch))

    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image']
        images_augmented = batch['image_augmented']
        b, c, h, w = images.size()
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w) 
        input_ = input_.cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)

        # aug_targets1 = torch.cat([a.unsqueeze(1) for a in batch['image_aug_labels'][4:7]], dim=1)
        # aug_targets2 = torch.cat([a.unsqueeze(1) for a in batch['image_augmented_aug_labels'][4:7]], dim=1)
        # aug_targets = torch.cat([aug_targets1, aug_targets2])
        # aug_targets = aug_targets.cuda(non_blocking=True)

        if augs_criterion is None:
            output = model(input_).view(b, 2, -1)
            loss = criterion(output)
            losses.update(loss.item())
        else:
            aug_targets = torch.cat([a.unsqueeze(1) for a in batch['aug_labels']], dim=1)
            # aug_targets = torch.cat([aug_targets, aug_targets])
            aug_targets = aug_targets.cuda(non_blocking=True)

            output, augs_output = model(input_)
            output = output.view(b, 2, -1)
            loss = criterion(output)
            losses.update(loss.item())
            augs_loss = augs_criterion(augs_output, aug_targets)
            augs_losses.update(augs_loss.item())
            loss = sum([loss, augs_loss])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)


def scan_train(train_loader, model, criterion, optimizer, epoch, update_cluster_head_only=False):
    """ 
    Train w/ SCAN-Loss
    """
    total_losses = AverageMeter('Total Loss', ':.4e')
    consistency_losses = AverageMeter('Consistency Loss', ':.4e')
    entropy_losses = AverageMeter('Entropy', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [total_losses, consistency_losses, entropy_losses],
        prefix="Epoch: [{}]".format(epoch))

    if update_cluster_head_only:
        model.eval() # No need to update BN
    else:
        model.train() # Update BN

    for i, batch in enumerate(train_loader):
        # Forward pass
        anchors = batch['anchor'].cuda(non_blocking=True)
        neighbors = batch['neighbor'].cuda(non_blocking=True)
       
        if update_cluster_head_only: # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                neighbors_features = model(neighbors, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='head')
            neighbors_output = model(neighbors_features, forward_pass='head')

        else: # Calculate gradient for backprop of complete network
            anchors_output = model(anchors)
            neighbors_output = model(neighbors)     

        # Loss for every head
        total_loss, consistency_loss, entropy_loss = [], [], []
        for anchors_output_subhead, neighbors_output_subhead in zip(anchors_output, neighbors_output):
            total_loss_, consistency_loss_, entropy_loss_ = criterion(anchors_output_subhead,
                                                                         neighbors_output_subhead)
            total_loss.append(total_loss_)
            consistency_loss.append(consistency_loss_)
            entropy_loss.append(entropy_loss_)

        # Register the mean loss and backprop the total loss to cover all subheads
        total_losses.update(np.mean([v.item() for v in total_loss]))
        consistency_losses.update(np.mean([v.item() for v in consistency_loss]))
        entropy_losses.update(np.mean([v.item() for v in entropy_loss]))

        total_loss = torch.sum(torch.stack(total_loss, dim=0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)


def selflabel_train(train_loader, model, criterion, optimizer, epoch, ema=None):
    """ 
    Self-labeling based on confident samples
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [losses],
                                prefix="Epoch: [{}]".format(epoch))
    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)

        with torch.no_grad(): 
            output = model(images)[0]
        output_augmented = model(images_augmented)[0]

        loss = criterion(output, output_augmented)
        losses.update(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None: # Apply EMA to update the weights of the network
            ema.update_params(model)
            ema.apply_shadow(model)
        
        if i % 25 == 0:
            progress.display(i)


def simclr_fine_tune_train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader),
        [losses, acc],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image']
        # images_augmented = batch['image_augmented']
        # b, c, h, w = images.size()
        # input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        # input_ = input_.view(-1, c, h, w)
        # input_ = input_.cuda(non_blocking=True)
        input_ = images.cuda(non_blocking=True)
        targets = np_to_tensor_safe(batch['target']).cuda(non_blocking=True)

        output = model(input_)
        loss = criterion(output, targets)
        losses.update(loss.item())
        acc1 = 100 * torch.mean(torch.eq(torch.argmax(output, dim=1), targets).float())
        acc.update(acc1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)