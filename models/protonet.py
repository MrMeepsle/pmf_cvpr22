import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        # bias & scale of cosine classifier
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)
        self.sigmoid = nn.Sigmoid()

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
        cls_scores = self.scale_cls * (cls_scores + self.bias)
        cls_scores = self.sigmoid(cls_scores)
        return cls_scores

    def predict_from_prototypes(self, supp_x, supp_y, x):
        """
        supp_x.shape = [B, nSupp, C, H, W]
        supp_y.shape = [B, nSupp]
        x.shape = [B, nQry, C, H, W]
        """
        num_classes = supp_y.max() + 1  # NOTE: assume B==1

        B, nSupp, C, H, W = supp_x.shape
        supp_f = self.backbone.forward(supp_x.view(-1, C, H, W))
        supp_f = supp_f.view(B, nSupp, -1)

        supp_y_1hot = F.one_hot(supp_y, num_classes).transpose(1, 2)  # B, nC, nSupp

        # B, nC, nSupp x B, nSupp, d = B, nC, d
        print(supp_y_1hot.shape, supp_f.shape, supp_x.shape)
        prototypes = torch.bmm(supp_y_1hot.float(), supp_f)
        prototypes = prototypes / supp_y_1hot.sum(dim=2, keepdim=True)  # NOTE: may div 0 if some classes got 0 images

        feat = self.backbone.forward(x.view(-1, C, H, W))
        feat = feat.view(B, x.shape[1], -1)  # B, nQry, d

        predictions = self.cos_classifier(prototypes, feat)  # B, nQry, nC
        return predictions

    def get_k_closest(self, support_class, support_images, support_labels, k=2):
        predictions = self.predict_from_prototypes(support_images, support_labels, support_class)
        print(predictions)

    def forward(self, support_class, support_class_label, support_tensor, support_labels, x_class, y_class, x_rest, y_rest):
        """
        supp_x.shape = [B, nSupp, C, H, W]
        supp_y.shape = [B, nSupp]
        x.shape = [B, nQry, C, H, W]
        """
        self.get_k_closest(x_class, support_tensor, support_labels)
        return self.predict_from_prototypes(supp_x, supp_y, x_rest)
