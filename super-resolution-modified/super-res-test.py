import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from models import get_net
from models.downsampler import Downsampler
from utils.common_utils import get_noise, get_params, optimize
from utils.sr_utils import tv_loss
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Load HR data
HR_tensor = torch.from_numpy(np.load("/home/nk1495/deep-image-prior/data/sr/label_500m_tensor.npy")).cuda()
HR_tensor = HR_tensor.unsqueeze(0).unsqueeze(0)

# Load LR data
LR_tensor = torch.from_numpy(np.load("/home/nk1495/deep-image-prior/data/sr/input_1000m_tensor.npy")).cuda()

# Network input
input_depth = 8
net_input = get_noise(input_depth, 'noise', (128, 128)).cuda().detach()

# DIP network # change back to reflection & skip_n33d=skip_n33u=128 num_scales = 5 if it doesn't work
net = get_net(input_depth, 'skip', pad='replication', 
              skip_n33d=128, skip_n33u=128, skip_n11=4,
              num_scales=5, upsample_mode='nearest',
              n_channels=1).cuda()

# Downsampler
downsampler = Downsampler(n_planes=8, factor=4, kernel_type='lanczos2', preserve_size=True).cuda()
downsampler.downsampler_ = downsampler.downsampler_.cuda()

# Loss setup
mse = nn.MSELoss().cuda()
LR_var = LR_tensor.detach().cuda()

# Optimization prep
reg_noise_std = 0.03
net_input_saved = net_input.cuda().detach().clone()
noise = net_input.cuda().detach().clone()

i = 0
def closure():
    global i, net_input
    if reg_noise_std > 0:
        net_input = net_input_saved + noise.normal_() * reg_noise_std

    out_HR = net(net_input)
    out_LR = downsampler(out_HR)

    total_loss = mse(out_LR, LR_var)
    total_loss.backward()

    print(f"Iteration {i:05d}, Loss: {total_loss.item():.6f}", end='\r')
    i += 1
    return total_loss

params = get_params('net', net, net_input)
optimize('adam', params, closure, LR=0.01, num_iter=2000)

# Evaluation
out_HR_final = net(net_input).detach()
out_LR_final = downsampler(out_HR_final)

psnr_lr = -10 * torch.log10(nn.functional.mse_loss(out_LR_final, LR_tensor)).item()
psnr_hr = -10 * torch.log10(nn.functional.mse_loss(out_HR_final, HR_tensor)).item()

print(f"\nPSNR (LR): {psnr_lr:.2f}")
print(f"PSNR (HR): {psnr_hr:.2f}")

# Visualization
plt.imshow(out_HR_final[0, 0].cpu(), cmap='gray')
plt.title("DIP Output HR")
plt.colorbar()
plt.show()

plt.imshow(HR_tensor[0, 0].cpu(), cmap='gray')
plt.title("Ground Truth HR")
plt.colorbar()
plt.show()
