import logging
import numpy as np
import math
from tqdm import tqdm
import torch
import os.path as osp
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy, forgetting
from utils.optim import GradualWarmupScheduler
from utils.loss_function import TripletLoss, _KD_loss



class iCaRL(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args['convnet_type'], args['method_args']['pretrained'])
        self.args = args

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_task = data_manager.nb_tasks
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        self._network.to(self._device)
        if self._cur_task == 0:
            args_type = "base_args"
        else:
            args_type = "train_args"

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # Loader
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                     source='train', mode='train', appendent=self._get_memory())

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')

        logging.info("train_dataset : {}, test_dataset : {}".format(len(train_dataset), len(test_dataset)))

        self.train_loader = DataLoader(train_dataset, batch_size=self.args[args_type]['batch_size'], shuffle=True,
                                       num_workers=self.args[args_type]['num_workers'])
        self.test_loader = DataLoader(test_dataset, batch_size=self.args[args_type]['batch_size'], shuffle=False,
                                  num_workers=self.args[args_type]['num_workers'])

        optimizer_conv = optim.SGD(self._network.convnet.parameters(),
                                   lr=self.args[args_type]['lr_conv'],
                                   momentum=self.args[args_type]['momentum'],
                                   weight_decay=self.args[args_type]['weight_decay'])
        scheduler_conv = optim.lr_scheduler.MultiStepLR(optimizer=optimizer_conv,
                                                        milestones=self.args[args_type]['milestones'],
                                                        gamma=self.args[args_type]['lr_decay'])
        optimizer_fc = optim.SGD(self._network.fc.parameters(),
                                 lr=self.args[args_type]['lr_fc'],
                                 momentum=self.args[args_type]['momentum'],
                                 weight_decay=self.args[args_type]['weight_decay'])
        scheduler_fc = optim.lr_scheduler.MultiStepLR(optimizer=optimizer_fc,
                                                      milestones=self.args[args_type]['milestones'],
                                                      gamma=self.args[args_type]['lr_decay'])
        self._update_representation(args_type, self.train_loader, self.test_loader, optimizer_conv, scheduler_conv, optimizer_fc, scheduler_fc)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)

    def _update_representation(self, args_type, train_loader, test_loader, optimizer_conv, scheduler_conv, optimizer_fc, scheduler_fc):
        path = osp.join(self.args['path'], self.args['time_str'], 'conv_{}_init_cls_{}_increment_{}_task_{}.pkl'.format(
            self.args['convnet_type'], self.args['init_cls'], self.args['increment'], self._cur_task) )

        if osp.exists(path):
            self._network.load_state_dict(torch.load(path))
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = 'Task [{}/{}], Test_accy {:.3f}'.format(self._cur_task + 1, self._total_task, test_acc)
            logging.info(info)
            return

        init_epoch = self.args[args_type]['epoch']

        if self.args[args_type]['warmup']:
            warmup_scheduler_conv = GradualWarmupScheduler(optimizer_conv, 1, self.args[args_type]['warmup_epoch'], scheduler_conv)
            warmup_scheduler_fc = GradualWarmupScheduler(optimizer_fc, 1, self.args[args_type]['warmup_epoch'], scheduler_fc)
            init_epoch += self.args[args_type]['warmup_epoch']

        if self.args['method_args']['TripletLoss']:
            Loss_Triplet = TripletLoss(margin=self.args['method_args']['margin'])

        print_fre = self.args['print_fre']

        for epoch in range(1, init_epoch + 1):
            self._network.train()
            losses = 0.
            loss1, loss2, loss3 = 0., 0., 0.
            correct, total = 0, 0

            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(torch.long).to(self._device)
                output = self._network(inputs)
                logits = output['logits']
                features = output['features']
                loss_clf = F.cross_entropy(logits, targets)

                loss = loss_clf
                if self.args['method_args']['TripletLoss']:
                    loss_triplet = Loss_Triplet(features, targets)
                    loss += self.args['method_args']['triplet_weight'] * loss_triplet

                if self._cur_task != 0:
                    loss_kd = _KD_loss(
                        logits[:, : self._known_classes],
                        self._old_network(inputs)["logits"],
                        self.args['method_args']['T'],
                    )

                    loss += self.args['method_args']['kd_weight'] * loss_kd

                optimizer_conv.zero_grad()
                optimizer_fc.zero_grad()
                loss.backward()
                optimizer_conv.step()
                optimizer_fc.step()

                losses += loss.item()
                loss1 += loss_clf.item()
                if self.args['method_args']['TripletLoss']:
                    loss2 += self.args['method_args']['triplet_weight'] * loss_triplet.item()
                if self._cur_task != 0:
                    loss3 += self.args['method_args']['kd_weight'] * loss_kd.item()

                # acc
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if self.args[args_type]['warmup']:
                warmup_scheduler_conv.step()
                warmup_scheduler_fc.step()
            else:
                scheduler_conv.step()
                scheduler_fc.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = 'Task [{}/{}], \tEpoch [{}/{}] \t=> \tconv lr = {:.6f}, \tfc lr = {:.6f}, \tloss = {:.3f}, \tloss_clf = {:.3f}, \tloss_triplet = {:.3f}, \tloss_kd = {:.3f}, \tTrain_accy = {:.3f}'.format(
                self._cur_task + 1, self._total_task, epoch, init_epoch,
                optimizer_conv.state_dict()['param_groups'][0]['lr'],
                optimizer_fc.state_dict()['param_groups'][0]['lr'],
                losses / len(train_loader), loss1 / len(train_loader),
                loss2 / len(train_loader), loss3 / len(train_loader),
                train_acc
            )
            if epoch % print_fre == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += ', \tTest_accy = {:.3f}'.format(test_acc)
            logging.info(info)

        if self.args['save']:
            torch.save(self._network.state_dict(), path)
        elif self.args.get('save_first', None) and self._cur_task == 0:
            torch.save(self._network.state_dict(), path)

