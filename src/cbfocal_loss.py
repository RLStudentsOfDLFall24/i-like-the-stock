import torch as th
import torch.nn as nn
import torch.nn.functional as f




# noinspection PyShadowingBuiltins
class FocalLoss(nn.Module):
    def __init__(self, class_counts, gamma=0.0):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = self.reweight(class_counts)

    def forward(self, input: th.tensor, target: th.tensor):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None
        # 1. Input should be logits, we need to softmax them
        preds = f.softmax(input, dim=1)

        # 2. Gather the target probs and weights for each label
        preds_target = preds.gather(1, target.reshape(-1, 1)).squeeze()
        weights = self.weight[target]
        log_term = th.log(preds_target)

        # 3. Compute the loss
        batch_losses = -weights * ((1 - preds_target) ** self.gamma) * log_term

        # 4. Loss will be the mean of the batch losses
        loss = th.mean(batch_losses)
        return loss

    @staticmethod
    def reweight(cls_num_list, beta=0.9999) -> th.Tensor:
        """
        Implement reweighting by effective numbers
        :param cls_num_list: a list containing # of samples of each class
        :param beta: hyper-parameter for reweighting, see paper for more details
        :return: A tensor containing the weights for each class
        """

        numerator = 1 - beta
        denominator = 1 - th.pow(beta, cls_num_list.clone().detach())

        e_n = numerator / (denominator + 1e-8)
        per_cls_weights = (e_n / e_n.sum()) * e_n.shape[0]

        return per_cls_weights