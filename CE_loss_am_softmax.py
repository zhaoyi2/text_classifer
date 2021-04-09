import torch
import torch.nn as nn
import torch.nn.functional as F


class AM_softmax(nn.Module):
    def __init__(self, m=0.3, s=20):
        super(AM_softmax, self).__init__()
        self.m = m
        self.s = s

    def forward(self, costh, target):

        target = target.view(-1, 1)

        costh_m = torch.zeros(costh.size()).scatter_(1, target, self.m)
        costh = costh - costh_m
        costh = self.s*costh
        if costh.is_cuda:
            costh = costh.cuda()

        output = torch.exp(costh)

        return output/output.sum(1).view(-1, 1)

class CE_loss(nn.Module):

    def __init__(self):
        super(CE_loss, self).__init__()
        self.am_softmax = AM_softmax()

    def forward(self,inputs, target):
        self.output = self.am_softmax(inputs, target)
        output = torch.log(self.output)
        target = F.one_hot(target)
        return -torch.sum(output*target)/inputs.size(0)


if __name__ == '__main__':

    inputs = torch.randn([2, 10], requires_grad=True)
    target = torch.tensor([0, 2])
    ce_loss = CE_loss(10, 3)
    loss = ce_loss(inputs, target)
    loss.backward()

    print(loss.detach().numpy())