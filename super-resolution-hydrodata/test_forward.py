import torch
import torch.nn as nn
from models import get_net
from models.downsampler import Downsampler

def test_forward_pass(device, pad_mode, num_scales=5):
    print(f"\nTesting on device={device} with pad='{pad_mode}', num_scales={num_scales}")
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    input_depth = 8
    net_input = torch.randn(1, input_depth, 128, 128, device=device)

    try:
        net = get_net(input_depth, 'skip', pad=pad_mode,
                      skip_n33d=64, skip_n33u=64, skip_n11=4,
                      num_scales=num_scales, upsample_mode='nearest',
                      n_channels=1).to(device)
        
        out = net(net_input)
        print("Forward pass succeeded.")
    except Exception as e:
        print(f"Forward pass FAILED: {e}")

if __name__ == "__main__":
    # 1. Test CPU (should always pass)
    test_forward_pass('cpu', pad_mode='reflection')

    # 2. Test GPU with reflection padding (likely to fail)
    if torch.cuda.is_available():
        test_forward_pass('cuda', pad_mode='reflection')

        # 3. Try replication padding on GPU (more stable)
        test_forward_pass('cuda', pad_mode='replication')

        # 4. Try zero padding on GPU (sometimes works)
        test_forward_pass('cuda', pad_mode='zero')

        # 5. Reduce num_scales to 3 and replication padding
        test_forward_pass('cuda', pad_mode='replication', num_scales=3)
