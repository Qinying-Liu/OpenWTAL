import torch
import torch.nn as nn
import numpy as np


class BaseModel(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(BaseModel, self).__init__()
        self.len_feature = len_feature
        self.num_classes = num_classes

        self.base_module = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.cls = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=self.num_classes, kernel_size=1, padding=0),
        )

        self.action_module_rgb = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0),
        )

        self.cls_rgb = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0)

        self.action_module_flow = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0),
        )

        self.cls_flow = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0)

        self.dropout = nn.Dropout(p=0.5)  # 0.5

    def forward(self, x):
        input = x.permute(0, 2, 1)

        emb_flow = self.action_module_flow(input[:, 1024:, :])
        emb_rgb = self.action_module_rgb(input[:, :1024, :])

        embedding_flow = emb_flow.permute(0, 2, 1)
        embedding_rgb = emb_rgb.permute(0, 2, 1)

        action_flow = torch.sigmoid(self.cls_flow(emb_flow))
        action_rgb = torch.sigmoid(self.cls_rgb(emb_rgb))

        emb = self.base_module(input)
        embedding = emb.permute(0, 2, 1)
        # emb = self.dropout(emb)
        cas = self.cls(emb).permute(0, 2, 1)
        actionness1 = cas.sum(dim=2)
        actionness1 = torch.sigmoid(actionness1)

        actionness2 = (action_flow + action_rgb) / 2
        actionness2 = actionness2.squeeze(1)

        return cas, action_flow, action_rgb, actionness1, actionness2, embedding, embedding_flow, embedding_rgb


class WTALModel(nn.Module):
    def __init__(self, config):
        super(WTALModel, self).__init__()
        self.len_feature = config.len_feature
        self.num_classes = config.num_classes

        self.actionness_module = BaseModel(self.len_feature, self.num_classes)

        self.softmax = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=2)

        self.r_C = 20
        self.r_I = 20

        self.dropout = nn.Dropout(p=0.6)

    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1)
        idx_topk = idx_DESC[:, :k]
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings

    def consistency_snippets_mining1(self, aness_bin1, aness_bin2, actionness1, embeddings, k_easy):
        x = aness_bin1 + aness_bin2
        select_idx_act = actionness1.new_tensor(np.where(x == 2, 1, 0))
        # print(torch.min(torch.sum(select_idx_act, dim=-1)))

        actionness_act = actionness1 * select_idx_act

        select_idx_bg = actionness1.new_tensor(np.where(x == 0, 1, 0))

        actionness_rev = torch.max(actionness1, dim=1, keepdim=True)[0] - actionness1
        actionness_bg = actionness_rev * select_idx_bg

        easy_act = self.select_topk_embeddings(actionness_act, embeddings, k_easy)
        easy_bkg = self.select_topk_embeddings(actionness_bg, embeddings, k_easy)

        return easy_act, easy_bkg

    def Inconsistency_snippets_mining1(self, aness_bin1, aness_bin2, actionness1, embeddings, k_hard):
        x = aness_bin1 + aness_bin2
        idx_region_inner = actionness1.new_tensor(np.where(x == 1, 1, 0))
        aness_region_inner = actionness1 * idx_region_inner
        hard_act = self.select_topk_embeddings(aness_region_inner, embeddings, k_hard)

        actionness_rev = torch.max(actionness1, dim=1, keepdim=True)[0] - actionness1
        aness_region_outer = actionness_rev * idx_region_inner
        hard_bkg = self.select_topk_embeddings(aness_region_outer, embeddings, k_hard)

        return hard_act, hard_bkg

    def forward(self, x):
        num_segments = x.shape[1]
        k_C = num_segments // self.r_C
        k_I = num_segments // self.r_I

        cas, action_flow, action_rgb, actionness1, actionness2, embedding, embedding_flow, embedding_rgb = self.actionness_module(
            x)

        if not self.training:
            return cas, action_flow, action_rgb

        aness_np1 = actionness1.cpu().detach().numpy()
        aness_median1 = np.median(aness_np1, 1, keepdims=True)
        aness_bin1 = np.where(aness_np1 > aness_median1, 1.0, 0.0)

        aness_np2 = actionness2.cpu().detach().numpy()
        aness_median2 = np.median(aness_np2, 1, keepdims=True)
        aness_bin2 = np.where(aness_np2 > aness_median2, 1.0, 0.0)

        # actionness = actionness1 + actionness2

        CA, CB = self.consistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding, k_C)
        IA, IB = self.Inconsistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding, k_I)

        CAr, CBr = self.consistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding_rgb, k_C)
        IAr, IBr = self.Inconsistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding_rgb, k_I)

        CAf, CBf = self.consistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding_flow, k_C)
        IAf, IBf = self.Inconsistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding_flow, k_I)

        contrast_pairs = {
            'CA': CA,
            'CB': CB,
            'IA': IA,
            'IB': IB
        }

        contrast_pairs_r = {
            'CA': CAr,
            'CB': CBr,
            'IA': IAr,
            'IB': IBr
        }

        contrast_pairs_f = {
            'CA': CAf,
            'CB': CBf,
            'IA': IAf,
            'IB': IBf
        }

        return cas, action_flow, action_rgb, contrast_pairs, contrast_pairs_r, contrast_pairs_f, actionness1, actionness2, aness_bin1, aness_bin2
