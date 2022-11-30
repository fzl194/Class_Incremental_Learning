import logging
import os
import numpy as np
import torch


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), 'Data length error.'
    all_acc = {}
    all_acc['total'] = np.around((y_pred == y_true).sum()*100 / len(y_true), decimals=2)

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = '{}-{}'.format(str(class_id).rjust(2, '0'), str(class_id+increment-1).rjust(2, '0'))
        all_acc[label] = np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2)

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc['old'] = 0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes),
                                                         decimals=2)

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc['new'] = np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2)

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)



def forgetting(accuracies):
    #accuracies = [[78],[65,78]]
    if len(accuracies) == 1:
        return 0.

    #当前模型的Acc
    now_accuracies = accuracies[-1]

    num_task = len(accuracies[-2])

    forgetting = 0.
    for task_id in range(num_task):
        max_task = 0.
        for task_accuracies in accuracies[:-1]:
            if len(task_accuracies) >= task_id + 1:
                max_task = max(max_task, task_accuracies[task_id])

        forgetting += max_task - now_accuracies[task_id]

    return forgetting / num_task


def forward(model, loader, device, path):
    model.eval()
    correct, total = 0, 0
    all_predicts = []
    all_features = []
    all_targets = []
    for i, (_, inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        predicts = torch.max(outputs['logits'], dim=1)[1].cpu()
        correct += (predicts == targets).sum()
        total += len(targets)

        predicts, targets, features = predicts.numpy(), targets.numpy(), outputs['features'].cpu().numpy()

        all_predicts.append(predicts)
        all_targets.append(targets)
        all_features.append(features)
    print("test acc = {:.3f}".format(100 * correct / total))
    all_predicts = np.concatenate(all_predicts).reshape(-1, 1)
    all_targets = np.concatenate(all_targets).reshape(-1, 1)
    all_features = np.concatenate(all_features).reshape(-1, 64)
    np.save(path + "_all_predicts.npy", all_predicts)
    np.save(path + "_all_targets.npy", all_targets)
    np.save(path + "_all_features.npy", all_features)


def new_old_forward(model, loader, old_class_number, device, epoch):
    model.eval()
    correct, total = 0, 0
    old_correct = 0
    old_total = 0
    new_correct = 0
    new_total = 0
    for i, (_, inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        predicts = torch.max(outputs['logits'], dim=1)[1].cpu()
        correct += (predicts == targets).sum().item()
        total += len(targets)

        old_correct += (predicts[targets < old_class_number] == targets[targets < old_class_number]).sum().item()
        old_total += (targets < old_class_number).sum().item()

        new_correct += (predicts[targets >= old_class_number] == targets[targets >= old_class_number]).sum().item()
        new_total += (targets >= old_class_number).sum().item()

    logging.info("epoch : {}".format(epoch))
    logging.info("correct : {}, total : {}, acc : {:.2f}".format(correct, total, correct / total))
    logging.info("old_correct : {}, old_total : {}, old_acc : {:.2f}".format(old_correct, old_total, old_correct / old_total))
    logging.info("new_correct : {}, new_total : {}, new_acc : {:.2f}".format(new_correct, new_total, new_correct / new_total))