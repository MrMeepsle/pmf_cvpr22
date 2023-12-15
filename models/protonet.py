import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        # maybe these need a different learning rate

        # bias & scale of classifier
        #####################################################################
        # P>M>F parameters
        self.b1 = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)
        self.w1 = nn.Parameter(torch.FloatTensor(1).fill_(1), requires_grad=True)

        self.softmax = nn.Softmax(dim=1)
        ####################################################################

        ####################################################################
        # ProtoProductNet parameters
        self.b2 = nn.Parameter(torch.FloatTensor(1).fill_(-8), requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)

        self.b3 = nn.Parameter(torch.FloatTensor(1).fill_(-1.1181), requires_grad=True)
        self.w3 = nn.Parameter(torch.FloatTensor(1).fill_(1.7963), requires_grad=True)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        ####################################################################

        # backbone
        self.backbone = backbone

    def cos_classifier(self, w, f):
        """
        w.shape = B, nC, d
        f.shape = B, M, d
        """
        f = F.normalize(f, p=2, dim=f.dim() - 1, eps=1e-12)
        w = F.normalize(w, p=2, dim=w.dim() - 1, eps=1e-12)

        cls_scores = f @ w.transpose(1, 2)  # B, M, nC
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

        cls_scores = self.cos_classifier(prototypes, feat)  # B, nQry, nC

        softmax_logits = self.w1 * (cls_scores + self.b1)
        softmax_probs = self.softmax(softmax_logits)

        tanh_logits = self.w2 * cls_scores + self.b2
        tanh_probs = self.tanh(tanh_logits)

        probabilities = self.sigmoid(self.w3 * (softmax_probs + tanh_probs) + self.b3)

        return probabilities, prototypes
