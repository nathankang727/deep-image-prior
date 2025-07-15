import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from models import get_net
from models.downsampler import Downsampler
from utils.common_utils import get_noise, get_params, optimize
from utils.sr_utils import tv_loss
import scipy
import time

start_time = time.time()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Load HR data
HR_tensor = torch.from_numpy(np.load("/home/nk1495/deep-image-prior/data/sr/label_500m_tensor.npy")).cuda()
# HR_tensor = HR_tensor.unsqueeze(0).unsqueeze(0)

# Load LR data
LR_tensor = torch.from_numpy(np.load("/home/nk1495/deep-image-prior/data/sr/input_1000m_tensor.npy")).cuda()
print(HR_tensor.shape)
print(LR_tensor.shape)

# Network input
input_depth = 8
net_input = get_noise(input_depth, 'noise', (64, 44)).cuda().detach()

# DIP network
net = get_net(input_depth, 'skip', pad='replication', 
              skip_n33d=128, skip_n33u=128, skip_n11=4,
              num_scales=3, upsample_mode='nearest',
              n_channels=8).cuda()

# Downsampler
downsampler = Downsampler(n_planes=8, factor=2, kernel_type='lanczos2', preserve_size=True).cuda()
downsampler.downsampler_ = downsampler.downsampler_.cuda()

# Loss setup
mse = nn.MSELoss().cuda()
LR_var = LR_tensor.detach().cuda()

# Optimization prep
reg_noise_std = 0.03
net_input_saved = net_input.detach().cuda().clone()
noise = net_input.detach().cuda().clone()

i = 0
iterations = []         # List of all iterations (x-axis for graph)
loss_values = []        # List of the loss values for each iteration (y-axis for graph)

best_loss = 999999999   # Best Loss value so far 
best_iter = 0           # Iteration where best loss value occurred
patience = 300          # Patience value
best_net_output = None  # Best network output value so far
early_stop_counter = 0  # Counter before patience

def closure():
    global i, net_input
    if reg_noise_std > 0:
        net_input = net_input_saved + noise.normal_() * reg_noise_std

    out_HR = net(net_input)
    out_LR = downsampler(out_HR)

    total_loss = mse(out_LR, LR_var)
    total_loss.backward()

    print(f"Iteration {i:05d}, Loss: {total_loss.item():.6f}", end='\r')
    
    # Patience/Early Stopping
    global best_loss, best_iter, patience, best_net_output, early_stop_counter
    
    if total_loss < best_loss:
        best_loss = total_loss
        best_iter = i
        early_stop_counter = 0
        best_net_output = out_HR
    else:
        early_stop_counter += 1
        if early_stop_counter > patience:
            print("\nPatience exceeded. Stopping at iteration " + str(i) + ".")
            return None
        
    iterations.append(i)
    loss_values.append(total_loss)
    i += 1
    
    return total_loss

params = get_params('net', net, net_input)
optimize('adam', params, closure, LR=0.001, num_iter=15000)

# Evaluation
out_HR_final = best_net_output.detach()                # shape: [1, 8, 64, 44]
out_LR_final = downsampler(out_HR_final).detach()      # shape: [1, 8, 64, 44]
out_HR_final_0 = out_HR_final[:, 0:1, :, :]            # shape: [1, 1, 64, 44]

# Save network output as numpy arrays
# torch.save(out_HR_final, "filename")
# torch.no_grad for disabling gradients
np.save("training_outputs/HR_500m_tensor_8_channels.npy", out_HR_final.cpu().numpy())
np.save("training_outputs/LR_1000m_tensor_8_channels.npy", out_LR_final.cpu().numpy())
np.save("training_outputs/HR_500m_tensor_1_channel.npy", out_HR_final_0.cpu().numpy())

# Visualization/Graphs
# Compare network output, label, and input.
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

im00 = axes[0][0].imshow(out_HR_final[0, 0].cpu(), cmap='gray')
axes[0][0].set_title("Best HR Output, 0th channel")
fig.colorbar(im00, ax=axes[0][0])

im01 = axes[0][1].imshow(HR_tensor.cpu(), cmap='gray')
axes[0][1].set_title("HR Label, 0th channel")
fig.colorbar(im01, ax=axes[0][1])

im10 = axes[1][0].imshow(out_LR_final[0, 0].cpu(), cmap='gray')
axes[1][0].set_title("Best LR Output, 0th channel")
fig.colorbar(im00, ax=axes[1][0])

im11 = axes[1][1].imshow(LR_tensor[0, 0].cpu(), cmap='gray')
axes[1][1].set_title("LR Input, 0th chanel")
fig.colorbar(im11, ax=axes[1][1])

plt.tight_layout()
plt.savefig("training_outputs/comparison.png", bbox_inches="tight")
plt.close()

# Loss graph
plt.plot(iterations, loss_values)
plt.savefig("training_outputs/Loss_Graph.png", bbox_inches="tight")
plt.close()

# Print best iteration & loss and total runtime.
print("\nBest iteration: " + str(best_iter))
print("Best loss: " + str(best_loss))
print("Total runtime: " + str(time.time() - start_time) + " seconds.")
