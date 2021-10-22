import torch
from torch import nn
import torch.nn.functional as F

from torchvision import datasets, models, transforms

# DCGAN loss
def loss_dcgan_dis(dis_fake, dis_real):
  L1 = torch.mean(F.softplus(-dis_real))
  L2 = torch.mean(F.softplus(dis_fake))
  return L1, L2


def loss_dcgan_gen(dis_fake):
  loss = torch.mean(F.softplus(-dis_fake))
  return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
  loss_real = torch.mean(F.relu(1. - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake
# def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
  # loss = torch.mean(F.relu(1. - dis_real))
  # loss += torch.mean(F.relu(1. + dis_fake))
  # return loss


def loss_hinge_gen(dis_fake):
  loss = -torch.mean(dis_fake)
  return loss

# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis


class EnsembleLosses(nn.Module):
  def __init__(self, ensemble, size_average=True):
    '''
    Ensemble is a list of neural networks all trained on the same dataset.
    '''
    super(EnsembleLosses, self).__init__()
    self.ensemble = ensemble
    self.size_average = size_average


  def get_softmax(self, input):
    '''
    Computes the softmax distribution from each component of the ensemble for a given input batch.
    '''

    # NOTE: The following code is for training ImageNet models only!!!
    input = (input * 0.5) + 0.5
    # Now normalize to the range required

    data_transforms = transforms.Compose([
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input = data_transforms(input)

    logits = []
    for classifier in self.ensemble:
      logits.append(classifier(input))
    logits = torch.stack(logits, dim=1)
    pt = F.softmax(logits, dim=2)
    return pt


  def entropy(self, pt):
    '''
    Computes the entropy of a given input. The input is of the form: N x K where N is the batch size
    and K is the number of classes.

    entropy = - \sum_{c=1}^K p_c \log p_c
    '''
    eps = 1e-40
    log_pt = torch.log(pt + eps)
    prod = pt * log_pt
    entropy = - torch.sum(prod, dim=(len(list(pt.shape)) - 1))
    return entropy


  def predictive_entropy(self, pt):
    '''
    Computes Predictive Entropy: Entropy of the expected predictive (softmax) distribution.
    '''

    # Computing expected softmax
    expected_pt = torch.mean(pt, dim=1)

    # Computing entropy of expected softmax
    pred_entropy = self.entropy(expected_pt)
    return pred_entropy


  def expected_entropy(self, pt):
    '''
    Computes the expectation of the entropies of individual components of the ensemble.
    '''

    entropy_ind = self.entropy(pt)
    expected_entropy = torch.mean(entropy_ind, dim=1)

    return expected_entropy
    

  def mutual_information(self, pt):
    '''
    Computes Mutual Information: (Predictive Entropy) - (Expectation of the individual entropies)
    '''

    # Compute predictive entropy
    pe = self.predictive_entropy(pt)

    # Compute entropy of inidividual components
    ee = self.expected_entropy(pt)
    
    mi = pe - ee
    return mi


  def n_cross_entropy(self, pt, target):
    '''
    Computes n-cross entropy. Given p_t as the probability assigned by a model to the ground truth correct class.
    nce = - 1/S \sum_{s=1}^S \log (1-p_st)
    '''
    pt_inv = (1 - pt)
    eps = 1e-40
    
    log_pt_inv = torch.log(pt_inv + eps)
    print (log_pt_inv.shape)
    log_pt_inv = - torch.gather(log_pt_inv, dim=2, index=target)
    nce = torch.mean(log_pt_inv, dim=1)
    
    return nce


  def forward(self, input, target, semantic=True):
    '''
    We always want to minimize expected_entropy (make the ensemble confident)
    if semantic = True, we want to minimize mutual_information, and n_cross_entropy (agreement between all models and all models are incorrect)
    if semantic = False, we want to maximize mutual_information (disagreement between all models)
    '''
    pt = self.get_softmax(input)
    mi = self.mutual_information(pt)

    loss = self.expected_entropy(pt)

    if semantic:
        nce = self.n_cross_entropy(pt, target)
        loss += mi + nce
    else:
        # loss = loss - mi
        loss = - mi

    if self.size_average: return loss.mean()
    else: return loss.sum()