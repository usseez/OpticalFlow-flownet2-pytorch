import torch
from collections import Counter

ckpt = torch.load("./weights/251114/451_FlowNet2C_checkpoint.pth.tar", map_location="cpu")
sd = ckpt["state_dict"]

dtypes = Counter(v.dtype for v in sd.values() if torch.is_tensor(v))
print(dtypes)