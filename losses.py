'''
Portions of this code copyright 2017, Clement Pinard
'''

# freda (todo) : adversarial loss 

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import torch.nn.functional as F_torch  # rename to avoid confusion with fundamental matrix

def sampson_distance(Fmat, x1, x2, eps=1e-8):
    # Fmat: (B,3,3), x1/x2: (B,H,W,3)
    Fx1   = torch.einsum('bij, bhwj -> bhwi', Fmat, x1)         # (B,H,W,3)
    Ftx2  = torch.einsum('bji, bhwj -> bhwi', Fmat, x2)         # (B,H,W,3)
    x2tFx1 = (x2 * Fx1).sum(dim=-1)                              # (B,H,W)

    denom = Fx1[..., 0]**2 + Fx1[..., 1]**2 + Ftx2[..., 0]**2 + Ftx2[..., 1]**2
    return (x2tFx1**2) / (denom + eps)

def epipolar_inlier_mask(flow, Fmat, thresh):
    # flow: (B,2,H,W), Fmat: (B,3,3)
    B, _, H, W = flow.shape
    device = flow.device

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    ones = torch.ones_like(xx)
    x1 = torch.stack([xx, yy, ones], dim=-1).float()            # (H,W,3)
    x1 = x1.unsqueeze(0).repeat(B, 1, 1, 1)                     # (B,H,W,3)

    x2 = x1.clone()
    x2[..., 0] = x2[..., 0] + flow[:, 0]                        # u + du
    x2[..., 1] = x2[..., 1] + flow[:, 1]                        # v + dv

    # 이미지 밖으로 나가면 invalid
    valid = (x2[...,0] >= 0) & (x2[...,0] <= (W - 1)) & (x2[...,1] >= 0) & (x2[...,1] <= (H - 1))

    d = sampson_distance(Fmat, x1, x2)                          # (B,H,W)
    inlier = (d < thresh) & valid
    return inlier.unsqueeze(1).float()                          # (B,1,H,W)

class EpipolarFilteredL1Loss(nn.Module):
    """
    target = [flow_gt, F] 를 기대.
    --loss EpipolarFilteredL1Loss --loss_epipolar_thresh 1.0 처럼 사용
    """
    def __init__(self, args, epipolar_thresh=1.0, min_inlier_ratio=0.05):
        super().__init__()
        self.epipolar_thresh = float(epipolar_thresh)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.loss_labels = ("EpiFilteredL1", "inlier_ratio")

    def forward(self, output, target):
        # output이 multi-scale list일 수도 있으니, 가장 첫 flow만 사용(필요하면 확장)
        pred_flow = output[0] if isinstance(output, (list, tuple)) else output

        flow_gt = target[0]
        Fmat = target[1]
        if Fmat.dim() == 2:
            Fmat = Fmat.unsqueeze(0)  # (1,3,3)

        # B 맞추기 (DataParallel scatter 시 보통 맞지만 안전장치)
        if Fmat.size(0) != pred_flow.size(0):
            Fmat = Fmat.repeat(pred_flow.size(0), 1, 1)

        mask = epipolar_inlier_mask(pred_flow, Fmat, self.epipolar_thresh)  # (B,1,H,W)
        inlier_ratio = mask.mean()

        # 너무 inlier가 적으면(예: F 추정 실패/비강체 장면) 마스크를 무시하는 안전장치
        if inlier_ratio.item() < self.min_inlier_ratio:
            mask = torch.ones_like(mask)
            inlier_ratio = mask.mean()

        l1 = (pred_flow - flow_gt).abs().sum(dim=1, keepdim=True)  # (B,1,H,W)
        loss = (mask * l1).sum() / (mask.sum() + 1e-6)

        return [loss, inlier_ratio]
    

def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue

class L1Loss(nn.Module):
    def __init__(self, args):
        super(L1Loss, self).__init__()
        self.args = args
        self.loss = L1()
        self.loss_labels = ['L1', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]

class L2Loss(nn.Module):
    def __init__(self, args):
        super(L2Loss, self).__init__()
        self.args = args
        self.loss = L2()
        self.loss_labels = ['L2', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]

class MultiScale(nn.Module):
    def __init__(self, args, startScale = 4, numScales = 5, l_weight= 0.32, norm= 'L1'):
        super(MultiScale,self).__init__()

        self.startScale = startScale
        self.numScales = numScales
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for scale in range(self.numScales)])
        self.args = args
        self.l_type = norm
        self.div_flow = 0.05
        assert(len(self.loss_weights) == self.numScales)

        if self.l_type == 'L1':
            self.loss = L1()
        else:
            self.loss = L2()

        self.multiScales = [nn.AvgPool2d(self.startScale * (2**scale), self.startScale * (2**scale)) for scale in range(self.numScales)]
        self.loss_labels = ['MultiScale-'+self.l_type, 'EPE'],

    def forward(self, output, target):
        lossvalue = 0
        epevalue = 0

        if type(output) is tuple:
            target = self.div_flow * target
            for i, output_ in enumerate(output):
                target_ = self.multiScales[i](target)
                epevalue += self.loss_weights[i]*EPE(output_, target_)
                lossvalue += self.loss_weights[i]*self.loss(output_, target_)
            return [lossvalue, epevalue]
        else:
            epevalue += EPE(output, target)
            lossvalue += self.loss(output, target)
            return  [lossvalue, epevalue]
        

#proxyLabelLoss

# losses.py

class ProxyLabelLoss(nn.Module):
    """
    Unsupervised loss for optical flow using:
      - pseudo labels (proxy_flow)
      - fundamental-matrix filtering to drop noisy labels

    Expected inputs:
      pred_flow:  (B, 2, H, W)
      target:     [proxy_flow, F]
                   proxy_flow: (B, 2, H, W)
                   F:          (B, 3, 3) or (B, 9) or (B, 9, 1, 1)
    """

    def __init__(
        self,
        args,
        loss_epipolar_thresh: float = 1.0,
        loss_proxy_weight: float = 1.0,
        loss_smooth_weight: float = 0.1,
    ):
        super().__init__()
        self.epipolar_thresh = loss_epipolar_thresh
        self.proxy_weight = loss_proxy_weight
        self.smooth_weight = loss_smooth_weight

        # the training script expects this
        self.loss_labels = ['total_proxy', 'proxy_epe', 'smooth']

    def _normalize_F(self, F):
        """
        Force F to shape (B, 3, 3).
        Accepts:
          (B, 3, 3),
          (B, 9),
          (B, 9, 1, 1)
        """
        if F.dim() == 2 and F.size(1) == 9:
            F = F.view(F.size(0), 3, 3)
        elif F.dim() == 4 and F.size(1) == 9:
            F = F.view(F.size(0), 3, 3)
        elif F.dim() == 4 and F.size(1) == 3 and F.size(2) == 3:
            F = F.view(F.size(0), 3, 3)
        # otherwise assume already (B, 3, 3)
        return F

    def _epipolar_mask(self, proxy_flow, F):
        """
        Compute an epipolar-consistency mask using the proxy flow and F.

        For each pixel x = (u, v, 1)^T, proxy correspondence:
          x' = (u + du, v + dv, 1)^T

        Epipolar line l' = F x, epipolar distance:
          d = |x'^T l'| / sqrt(a^2 + b^2)
        Keep pixels with d < threshold.
        """
        B, _, H, W = proxy_flow.shape
        device = proxy_flow.device
        dtype = proxy_flow.dtype

        F = self._normalize_F(F)

        # Pixel coordinates grid
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij'
        )  # (H, W)
        ones = torch.ones_like(xs)
        x = torch.stack([xs, ys, ones], dim=0)          # (3, H, W)
        x = x.view(1, 3, -1).repeat(B, 1, 1)            # (B, 3, N)

        # x' = x + flow
        flow_flat = proxy_flow.view(B, 2, -1)           # (B, 2, N)
        x_prime = x.clone()
        x_prime[:, 0, :] = x_prime[:, 0, :] + flow_flat[:, 0, :]
        x_prime[:, 1, :] = x_prime[:, 1, :] + flow_flat[:, 1, :]

        # Epipolar line l' = F x
        l = torch.bmm(F, x)                             # (B, 3, N), l = (a, b, c)^T

        # distance from x' to l'
        num = (x_prime * l).sum(dim=1).abs()            # (B, N)
        den = torch.sqrt(l[:, 0, :] ** 2 + l[:, 1, :] ** 2 + 1e-8)
        d = num / den                                   # (B, N)

        mask = d < self.epipolar_thresh                 # (B, N)
        mask = mask.view(B, 1, H, W)                    # (B, 1, H, W)
        return mask

    def _smoothness_loss(self, flow):
        """
        First-order smoothness penalty on the flow field.
        """
        dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
        dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
        return dx.mean() + dy.mean()

    def forward(self, pred_flow, target):
        """
        pred_flow: (B, 2, H, W)
        target:    [proxy_flow, F]
        """
        # unpack
        if not isinstance(target, (list, tuple)) or len(target) != 2:
            raise ValueError(
                "ProxyLabelLoss expects target to be [proxy_flow, F]. "
                f"Got type {type(target)} with length {len(target) if isinstance(target, (list, tuple)) else 'N/A'}"
            )

        proxy_flow, F = target
        B, C, H, W = pred_flow.shape

        # EPE (end-point error) between pred and proxy
        diff = pred_flow - proxy_flow
        epe = torch.sqrt(torch.sum(diff ** 2, dim=1))   # (B, H, W)

        # epipolar mask (1 = geometrically consistent)
        epi_mask = self._epipolar_mask(proxy_flow, F)   # (B, 1, H, W)
        epi_mask_flat = epi_mask.squeeze(1)             # (B, H, W)

        if epi_mask_flat.sum() > 0:
            proxy_loss = (epe * epi_mask_flat).sum() / epi_mask_flat.sum()
        else:
            # F failed badly, just use all pixels as fallback
            proxy_loss = epe.mean()

        smooth_loss = self._smoothness_loss(pred_flow)

        total_loss = self.proxy_weight * proxy_loss + self.smooth_weight * smooth_loss

        # training loop expects a list of tensors
        return [total_loss, proxy_loss, smooth_loss]
