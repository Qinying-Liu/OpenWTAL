import os
import torch
import random
import json
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F

from inference import inference
from utils import misc_utils
from dataset.thumos_features import ThumosFeature
from models.asl_model import WTALModel
from utils.loss import CrossEntropyLoss, GeneralizedCE
from config.config_asl import Config, parse_args

np.set_printoptions(formatter={'float_kind': "{:.2f}".format})


def get_dataloaders(config):
    train_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='train',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='random', supervision='strong'),
        batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers, drop_last=True)

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
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr,
                                          betas=(0.9, 0.999), weight_decay=0.0005)
        self.criterion = CrossEntropyLoss()
        self.Lgce = GeneralizedCE(q=self.config.q_val)

        # placeholder of clustering classification results
        self.cls_gt = torch.ones((config.num_clusters, 2), dtype=torch.float).cuda() / config.num_clusters

        # parameters
        self.best_mAP = -1  # init
        self.step = 0
        self.total_loss_per_epoch = 0

    def test(self):
        self.net.eval()

        with torch.no_grad():
            model_filename = "best_model.pkl"
            self.config.model_file = os.path.join(self.config.model_path, model_filename)
            _mean_ap, test_acc, _, ap = inference(self.net, self.config, self.test_loader, cls_gt=cls_gt,
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
        clu_agnostic_gt = []
        for b in range(batch_size):
            label_indices_b = torch.nonzero(label[b, :])[:, 0]
            topk_indices_b = topk_indices[b, :, label_indices_b]  # topk, num_actions
            clu_agnostic_gt_b = torch.zeros((1, 1, self.config.num_segments)).cuda()

            # positive examples
            for gt_i in range(len(label_indices_b)):
                clu_agnostic_gt_b[0, 0, topk_indices_b[:, gt_i]] = 1
            clu_agnostic_gt.append(clu_agnostic_gt_b)

        return torch.cat(clu_agnostic_gt, dim=0)  # B, 1, num_segments

    def evaluate(self, epoch=0):
        if self.step % self.config.detection_inf_step == 0:
            self.total_loss_per_epoch /= self.config.detection_inf_step
            with torch.no_grad():
                self.net = self.net.eval()
                mean_ap, test_acc, final_res, ap = inference(self.net, self.config, self.test_loader,
                                                             cls_gt=self.cls_gt, model_file=None)
                self.net = self.net.train()

            if mean_ap > self.best_mAP:
                self.best_mAP = mean_ap
                torch.save(self.net.state_dict(), os.path.join(self.config.model_path, "best_model.pkl"))
                np.save(os.path.join(self.config.model_path, 'cls_gt.npy'), self.cls_gt.detach().cpu().numpy())

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

    def train(self):
        for epoch in range(self.config.num_epochs):

            for _data, _label, _, _, _ in self.train_loader:
                batch_size = _data.shape[0]
                _data, _label = _data.cuda(), _label.cuda()
                self.optimizer.zero_grad()

                (cas_flow, cas_rgb), (att_flow, att_rgb) = self.net(_data)
                # cas_flow, cas_rgb: T-CAS (B, T, K)
                # att_flow, att_rgb: attention weights (B, 1, T)

                #################### baseline ####################
                # multiple instance learning
                combined_cas = misc_utils.instance_selection_function(
                    cas_flow.softmax(-1) + cas_rgb.softmax(-1),
                    att_flow.permute(0, 2, 1),
                    att_rgb.permute(0, 2, 1))
                _, topk_indices = torch.topk(combined_cas, self.config.num_segments // 8, dim=1)
                cas_top_flow = torch.mean(torch.gather(cas_flow, 1, topk_indices), dim=1)
                cas_top_rgb = torch.mean(torch.gather(cas_rgb, 1, topk_indices), dim=1)

                # video classification loss
                vid_loss = 0.5 * self.criterion(cas_top_flow, _label) + 0.5 * self.criterion(cas_top_rgb, _label)
                cost = vid_loss

                # compute foreground/background pseudo-labels
                att_gt = self.calculate_pesudo_target(batch_size, _label, topk_indices)

                # foreground/background classification loss
                att_loss_flow = self.Lgce(att_flow.squeeze(1), att_gt.squeeze(1))
                att_loss_rgb = self.Lgce(att_rgb.squeeze(1), att_gt.squeeze(1))
                cost += att_loss_flow + att_loss_rgb

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
