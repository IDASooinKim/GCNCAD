import torch
import numpy as np
from torch.nn import CrossEntropyLoss


class DiceLoss(torch.nn.Module):

        def __init__(self, 
                     smooth:float=1e-5
                ):

                super(DiceLoss, self).__init__()
                self.smooth = smooth

        
        def forward(self, 
                    label_pred:torch.Tensor, label_gt:torch.Tensor
                ):
  
                label_gt = torch.nn.functional.one_hot(label_gt, num_classes=8) #.transpose(1, 3).transpose(2,3)
                label_gt = torch.swapaxes(label_gt, 1, 3)
                label_gt = torch.swapaxes(label_gt, 2, 3)

                label_pred = torch.sigmoid(label_pred)
         
                iflat = label_pred.contiguous().view(-1)
                tflat = label_gt.contiguous().view(-1)
                intersection = (iflat * tflat).sum()

                A_sum = torch.sum(tflat * iflat)
                B_sum = torch.sum(tflat * tflat)
                
                return 1 - ((2. * intersection + self.smooth) / (A_sum + B_sum + self.smooth) )


class CombineLoss(torch.nn.Module):


        def __init__(self, 
                     smooth:float=1e-5, regular:float=0.3, 
                     ignore_index:int=-1
                ):

                super(CombineLoss, self).__init__()
                weights_map = torch.tensor([0.76687459, 0.96341123, 0.95000466, 0.99050368, 0.97346616, 0.58448934, 0.77125035, 0.05]).to('cuda')
                self.criterion_classification = CrossEntropyLoss(ignore_index=ignore_index, weight=weights_map)
                self.criterion_reconst = CrossEntropyLoss(ignore_index=ignore_index, weight=weights_map)
                # self.criterion_reconst = DiceLoss(smooth=smooth)
                self.regular = regular
                #self.weighted_class = WeightedClass()

        def forward(self,
                        class_pred:torch.Tensor, class_gt:torch.Tensor,
                        label_pred:torch.Tensor, label_gt:torch.Tensor
                ):
                
                class_pred = torch.swapaxes(class_pred, 2, 1)
                class_gt = class_gt.type(torch.long)
                
                classification_loss = self.criterion_classification(class_pred, class_gt)
    
                reconst_loss = self.criterion_reconst(label_pred, label_gt)

                #total_loss = (1-self.regular)*classification_loss + self.regular*reconst_loss
                total_loss = classification_loss + reconst_loss
                return total_loss
