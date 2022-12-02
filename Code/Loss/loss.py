"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

from __future__ import print_function, division
import torch.autograd as autograd
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)  # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore:  # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)]  # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                    for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

def uniform_sample_loss( gt_occ, gt, out, sample_ratio=1.0,k=5):
    if sample_ratio<1.0:
        with torch.no_grad():
            p=int(k/2)
            weight_occ=F.max_pool3d(gt_occ,kernel_size=k,stride=1,padding=p)
            weight_occ=F.avg_pool3d(weight_occ,kernel_size=k,stride=1,padding=p)
            weight_occ[weight_occ==0]=sample_ratio
        weight_occ = torch.where(torch.rand_like(weight_occ) < weight_occ, torch.ones_like(weight_occ), torch.zeros_like(weight_occ))
        weight_occ+=gt_occ
        weight_occ[weight_occ>0]=1
        out=out*weight_occ
        return l1_loss(gt-out)/max(torch.sum(weight_occ),1.0)
    else:
        # gt_occ[:,:,0:56,...]=gt_occ[:,:,0:56,...]*3
        loss_weight=gt_occ.clone()
        loss_weight[:,:,0:65,...]=loss_weight[:,:,0:65,...]*2
        return l1_loss((gt - out)*loss_weight) / (max(torch.sum(loss_weight), 1.0))


def probability_sample_loss(gt_occ,gt,out,sample_ratio=1.0,k=3):
    with torch.no_grad():
        p = int(k / 2)
        weight_occ = F.max_pool3d(gt_occ, kernel_size=k, stride=1, padding=p)
        weight_occ = F.avg_pool3d(weight_occ, kernel_size=k, stride=1, padding=p)
    loss_weight=(1-weight_occ.clone())+gt_occ

    all_occ=weight_occ.clone()
    all_occ[all_occ>0]=1
    all_occ[all_occ==0]=sample_ratio
    all_occ=torch.where(torch.rand_like(all_occ) < all_occ, torch.ones_like(all_occ), torch.zeros_like(all_occ))
    weight_occ[weight_occ<1]=0
    final_sample=all_occ-weight_occ+gt_occ
    out=out*final_sample

    #loss_weight=final_sample.clone()

    loss_weight=loss_weight*final_sample
    loss_weight[:,:,0:48,...]=loss_weight[:,:,0:48,...]*3
    return l1_loss((gt-out)*loss_weight)/max(torch.sum(loss_weight),1.0)






def l1_loss( x):
    return torch.sum(torch.abs(x))



def cosLoss(out,gt,mask):
    # norm=torch.norm((gt+1)/2,2,dim=1,keepdim=True)
    # print(gt[:,:,0,0])
    # black_regin=torch.where(norm>1e-1,torch.ones_like(gt[:,0:1,...]),torch.zeros_like(gt[:,0:1,...])).detach()
    # black_regin=(gt[:,0:1,...]!=).type(torch.float)
    # print(torch.max(black_regin))
    # print(torch.max(gt[:,0:1,...]))
    ori_regin=mask

    # save_image(ori_regin,'test.png')
    # print(torch.max(ori_regin))
    L1_loss=(3*l1_loss((gt-out)*ori_regin))/(max(torch.sum(ori_regin),1.0))
    # L1_loss=(3*l1_loss((gt-out)))/(512*512)
    # print(L1_loss)

    cosin=torch.cosine_similarity(out,gt,dim=1)
    loss=(1-cosin)*ori_regin[:,0,...]
    cosin_loss=torch.where(loss<1,torch.zeros_like(loss),loss)
    cosin_loss=torch.sum(cosin_loss)/(max(torch.sum(ori_regin),1.0))
    # print(cosin_loss)
    # return L1_loss
    return L1_loss




def binary_cross_entropy(logits,labels,weights):
    """
       NOTE: the range of the labels is {0, 1}
           r = gamma : to balance the training !!!
           z = labels
           x = logits
           loss =
           r * z * -log(sigmoid(x)) + (1 - r) * (1 - z) * -log(1 - sigmoid(x))
           = r * z * log(1 + exp(-x)) + (1 - r) * (1 - z) * (x + log(1 + exp(-x))
           = (1 - z - r + r * z) * x + (1 - z - r + 2 * r * z) * log(1 + exp(-x))
           set a = 1 - z - r
           set b = r * z
           (a + b) * x + (a + 2b) * log(1 + exp(-x))
           when x < 0, to prevent overflow
           (a + 2b) * log(1 + exp(-x)) = (a + 2b) * (-x + log(exp(x) + 1))

           when x < 0
           = - b * x + (a + 2b) * log(1 + exp(x))
           when x > 0
           = (a + b) * x + (a + 2b) * log(1 + exp(-x))

           to avoid overflow and enforce stability:
           = max(x, 0) * a + b * abs(x) + (a + 2b) * log(1 + exp(-abs(x))
       """
    assert logits.size()==labels.size()==weights.size(), "logits and labels must have the same shape{} vs {}".format(logits.size(),labels.size())

    # labels=labels.detach()
    # logits=logits.detach()

    gamma=0.85
    a=(1-gamma-labels)
    b=gamma*labels
    zeros=torch.zeros_like(logits)
    cond=(logits>=zeros)
    relu_logits=torch.where(cond,logits,zeros)
    neg_abs_logits=torch.where(cond,-logits,logits)
    pos_abs_logits=torch.where(cond,logits,-logits)
    loss=a*relu_logits+b*pos_abs_logits+(a+2*b)*torch.log1p(torch.exp(neg_abs_logits))
    loss*=weights

    return torch.sum(loss)/torch.sum(weights)



def compute_gradient_penalty(net, real_samples, fake_samples,lambda_gp=10.0):
  """Calculates the gradient penalty loss for WGAN GP"""
  # Random weight term for interpolation between real and fake samples
  alpha = torch.randn(real_samples.size(0), 1, 1, 1,1).cuda()
  # Get random interpolation between real and fake samples
  interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
  d_interpolates = net(interpolates)[-1]
  fake = torch.full((real_samples.size(0), ), 1).cuda()
  # Get gradient w.r.t. interpolates
  gradients = autograd.grad(
    outputs=d_interpolates,
    inputs=interpolates,
    grad_outputs=fake,
    create_graph=True,
    retain_graph=True,
    only_inputs=True,
  )[0]
  gradients = gradients.view(gradients.size(0), -1)
  gradient_penaltys = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
  return gradient_penaltys


def L1Loss(out,gt,weight):
    return l1_loss((gt - out) * weight) / max(torch.sum(weight), 1.0)