import os
import torch
import random
import json
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from inference import inference
from utils import misc_utils
from dataset.thumos_features import ThumosFeature
from utils.loss import CrossEntropyLoss, GeneralizedCE, ContrastiveLoss
from config.config_aicl import Config, parse_args
from models.aicl_model import WTALModel

np.set_printoptions(formatter={'float_kind': "{:.2f}".format})


def get_dataloaders(config):
    train_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='train',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='random', supervision='strong'),
        batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers)

    test_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='test',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='uniform', supervision='strong'),
        batch_size=1,
        shuffle=False, num_workers=config.num_workers)

    return train_loader, test_loader


def set_seed(config):
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.deterministic = True
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = False


class Trainer():
    def __init__(self, config):
        # config
        self.config = config

        # network
        self.net = WTALModel(config)
        self.net = self.net.cuda()

        # data
        self.train_loader, self.test_loader = get_dataloaders(self.config)

        # loss, optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr, betas=(0.9, 0.999),
                                          weight_decay=0.0005)
        self.criterion = CrossEntropyLoss()
        self.Lgce = GeneralizedCE(q=self.config.q_val)

        # parameters
        self.best_mAP = -1  # init
        self.step = 0
        self.total_loss_per_epoch = 0

    def test(self):
        self.net.eval()

        with torch.no_grad():
            model_filename = "best_model.pkl"
            self.config.model_file = os.path.join(self.config.model_path, model_filename)
            _mean_ap, test_acc, _, ap = inference(self.net, self.config, self.test_loader,
                                                  model_file=self.config.model_file)
            print("cls_acc={:.5f} map={:.5f}".format(test_acc * 100, _mean_ap * 100))

            ap = ap[:7]
            iou = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

            sum = 0
            count = 0
            for item in list(zip(iou, ap)):
                print('Detection map @ %f = %f' % (item[0], item[1]))
                if count < 7:
                    sum = sum + item[1]
                    count += 1
            print('Detection Avg map[0.1:0.5] = %f' % (np.sum(ap[:5]) / 5))
            print('Detection Avg map[0.3:0.7] = %f' % (np.sum(ap[2:]) / 5))
            print('Detection Avg map[0.1:0.7] = %f' % (np.sum(ap) / 7))

    def calculate_pesudo_target(self, batch_size, label, topk_indices):
        cls_agnostic_gt = []
        cls_agnostic_neg_gt = []
        for b in range(batch_size):
            label_indices_b = torch.nonzero(label[b, :])[:, 0]
            topk_indices_b = topk_indices[b, :, label_indices_b]  # topk, num_actions
            cls_agnostic_gt_b = torch.zeros((1, 1, self.config.num_segments)).cuda()

            # positive examples
            for gt_i in range(len(label_indices_b)):
                cls_agnostic_gt_b[0, 0, topk_indices_b[:, gt_i]] = 1
            cls_agnostic_gt.append(cls_agnostic_gt_b)

        return torch.cat(cls_agnostic_gt, dim=0)  # B, 1, num_segments

    def calculate_all_losses1(self, contrast_pairs, contrast_pairs_r, contrast_pairs_f, cas_top, label, action_flow,
                              action_rgb, cls_agnostic_gt, actionness1, actionness2):
        self.contrastive_criterion = ContrastiveLoss()
        loss_contrastive = self.contrastive_criterion(contrast_pairs) + self.contrastive_criterion(
            contrast_pairs_r) + self.contrastive_criterion(contrast_pairs_f)

        base_loss = self.criterion(cas_top, label)
        class_agnostic_loss = self.Lgce(action_flow.squeeze(1), cls_agnostic_gt.squeeze(1)) + self.Lgce(
            action_rgb.squeeze(1), cls_agnostic_gt.squeeze(1))

        modality_consistent_loss = 0.5 * F.mse_loss(action_flow, action_rgb) + 0.5 * F.mse_loss(action_rgb, action_flow)
        action_consistent_loss = 0.5 * F.mse_loss(actionness1, actionness2) + 0.5 * F.mse_loss(actionness2, actionness1)

        cost = base_loss + class_agnostic_loss + 5 * modality_consistent_loss + 0.01 * loss_contrastive + 0.1 * action_consistent_loss

        return cost

    def evaluate(self, epoch=0):
        if self.step % self.config.detection_inf_step == 0:
            self.total_loss_per_epoch /= self.config.detection_inf_step
            with torch.no_grad():
                self.net = self.net.eval()
                mean_ap, test_acc, final_res, ap = inference(self.net, self.config, self.test_loader,
                                                             model_file=None)
                self.net = self.net.train()

            if mean_ap > self.best_mAP:
                self.best_mAP = mean_ap
                torch.save(self.net.state_dict(), os.path.join(self.config.model_path, "best_model.pkl"))

                f_path = os.path.join(self.config.model_path, 'best.txt')
                with open(f_path, 'w') as f:
                    iou = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    string_to_write = "epoch: {:03d}  mAP: {:.2f}".format(epoch, mean_ap * 100)
                    f.write(string_to_write + '\n')
                    f.flush()

                    sum = 0
                    count = 0
                    for item in list(zip(iou, ap)):
                        sum = sum + item[1]
                        count += 1
                        string_to_write = 'Detection map @ %0.1f = %0.2f' % (item[0], item[1] * 100)
                        f.write(string_to_write + '\n')
                        f.flush()
                    string_to_write = 'Detection Avg map[0.1:0.5] = %0.2f' % (np.sum(ap[:5]) * 100 / 5)
                    f.write(string_to_write + '\n')
                    f.flush()
                    string_to_write = 'Detection Avg map[0.3:0.7] = %0.2f' % (np.sum(ap[2:7]) * 100 / 5)
                    f.write(string_to_write + '\n')
                    f.flush()
                    string_to_write = 'Detection Avg map[0.1:0.7] = %0.2f' % (np.sum(ap[:7]) * 100 / 7)
                    f.write(string_to_write + '\n')
                    f.flush()

                json_path = os.path.join(self.config.model_path, 'best_result.json')

                with open(json_path, 'w') as f:
                    json.dump(final_res, f)
                    f.close()

            print("epoch={:5d}  step={:5d}  Loss={:.4f}  cls_acc={:5.2f}  best_map={:5.2f}".format(
                epoch, self.step, self.total_loss_per_epoch, test_acc * 100, self.best_mAP * 100))

            self.total_loss_per_epoch = 0

    def forward_pass(self, _data):
        cas, action_flow, action_rgb, contrast_pairs, contrast_pairs_r, contrast_pairs_f, actionness1, actionness2, aness_bin1, aness_bin2 = self.net(
            _data)

        combined_cas = misc_utils.instance_selection_function(torch.softmax(cas.detach(), -1),
                                                              action_flow.permute(0, 2, 1).detach(),
                                                              action_rgb.permute(0, 2, 1))

        _, topk_indices = torch.topk(combined_cas, self.config.num_segments // 8, dim=1)
        # _, topk_indices1 = torch.topk(combined_cas, r, dim=1)
        cas_top = torch.mean(torch.gather(cas, 1, topk_indices), dim=1)

        return cas_top, topk_indices, action_flow, action_rgb, contrast_pairs, contrast_pairs_r, contrast_pairs_f, actionness1, actionness2, aness_bin1, aness_bin2

    def train(self):
        # training
        for epoch in range(self.config.num_epochs):

            for _data, _label, temp_anno, _, _ in self.train_loader:
                batch_size = _data.shape[0]
                _data, _label = _data.cuda(), _label.cuda()
                self.optimizer.zero_grad()

                # forward pass
                cas_top, topk_indices, action_flow, action_rgb, contrast_pairs, contrast_pairs_r, contrast_pairs_f, actionness1, actionness2, aness_bin1, aness_bin2 = self.forward_pass(
                    _data)

                # calcualte pseudo target
                cls_agnostic_gt = self.calculate_pesudo_target(batch_size, _label, topk_indices)

                # losses
                cost = self.calculate_all_losses1(contrast_pairs, contrast_pairs_r, contrast_pairs_f, cas_top, _label,
                                                  action_flow, action_rgb, cls_agnostic_gt, actionness1, actionness2)

                cost.backward()
                self.optimizer.step()

                self.total_loss_per_epoch += cost.cpu().item()
                self.step += 1

                # evaluation
                self.evaluate(epoch=epoch)


def main():
    args = parse_args()
    config = Config(args)
    set_seed(config)

    trainer = Trainer(config)

    if args.inference_only:
        trainer.test()
    else:
        trainer.train()


if __name__ == '__main__':
    main()
