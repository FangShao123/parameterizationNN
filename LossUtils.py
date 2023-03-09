import torch.nn as nn
import torch


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

# PP参数化方式计算损失
class PP_loss(nn.Module):
    def __init__(self):
        super(PP_loss, self).__init__()
        self.mse = nn.MSELoss()
    def forward(self, X, Ri):
        Ripow = torch.pow(10, Ri)
        Kt = 5e-3 / (1 + 5 * Ripow)**2 + 1e-4
        return self.mse(X, Kt)

class R2_loss(nn.Module):
    def __init__(self):
        super(R2_loss, self).__init__()
    def forward(self, X, Y):
        a = torch.sum((Y - X) ** 2)
        v_m = torch.mean(Y)
        b = torch.sum((Y - v_m) ** 2)
        R2 = 1 - a / b
        return R2
