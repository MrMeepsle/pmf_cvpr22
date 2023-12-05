import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoNet(nn.Module):
    def __init__(self, backbone, sigmoid=True):
        super().__init__()

        # bias & scale of cosine classifier
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)

        # backbone
        self.backbone = backbone
        self.sigmoid = None
        if sigmoid:
            self.sigmoid = torch.nn.Sigmoid()

    def cos_classifier(self, w, f):
        """
        w.shape = B, nC, d
        f.shape = B, M, d
        """
        f = F.normalize(f, p=2, dim=f.dim() - 1, eps=1e-12)
        w = F.normalize(w, p=2, dim=w.dim() - 1, eps=1e-12)

        cls_scores = f @ w.transpose(1, 2)  # B, M, nC
        cls_scores = self.scale_cls * (cls_scores + self.bias)
        if self.sigmoid is not None:
            cls_scores = self.sigmoid(cls_scores)

        return cls_scores

    def forward(self, supp_x, supp_y, x):
        """
        supp_x.shape = [B, nSupp, C, H, W]
        supp_y.shape = [B, nSupp]
        x.shape = [B, nQry, C, H, W]
        """
        B, nSupp, C, H, W = supp_x.shape

        supp_f = self.backbone.forward(supp_x.view(-1, C, H, W))
        supp_f = supp_f.view(B, nSupp, -1)
        supp_y = supp_y.transpose(1, 2)

        # B, nC, nSupp x B, nSupp, d = B, nC, d
        prototypes = torch.bmm(supp_y.float(), supp_f)
        prototypes = prototypes / supp_y.sum(dim=2, keepdim=True)  # NOTE: may div 0 if some classes got 0 images

        feat = self.backbone.forward(x.view(-1, C, H, W))
        feat = feat.view(B, x.shape[1], -1)  # B, nQry, d

        logits = self.cos_classifier(prototypes, feat)  # B, nQry, nC
        return logits, prototypes
