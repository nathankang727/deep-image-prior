import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models import get_net
from models.downsampler import Downsampler
from models.common_utils import get_noise, get_params, optimize
from models.sr_utils import tv_loss

# Create synthetic HR image
HR_tensor = torch.rand(1, 1, 128, 128).cuda()  # [B, C, H, W]

# Create synthetic LR image
LR_tensor = nn.functional.interpolate(HR_tensor, scale_factor=1/4, mode='nearest')  # or bicubic

# Network input
input_depth = 8
net_input = get_noise(input_depth, 'noise', (128, 128)).cuda().detach()

# DIP network
net = get_net(input_depth, 'skip', pad='reflection',
              skip_n33d=128, skip_n33u=128, skip_n11=4,
              num_scales=5, upsample_mode='nearest',
              n_channels=1).cuda()

# Downsampler
downsampler = Downsampler(n_planes=1, factor=4, kernel_type='lanczos2', preserve_size=True).cuda()

# Loss setup
mse = nn.MSELoss().cuda()
LR_var = LR_tensor.detach()

# Optimization prep
reg_noise_std = 0.03
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

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
