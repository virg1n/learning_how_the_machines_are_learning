import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import CosineAnnealingLR

max_steps = 150
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1024
num_epochs = 200

print(device)

def cosine_beta_schedule(timesteps, s=0.008, device=device):
    t = torch.linspace(0, timesteps, timesteps + 1, device=device, dtype=torch.float64)
    f = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(1e-8, 0.999).float()

betas = cosine_beta_schedule(max_steps, device=device)
alphas = 1 - betas

def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1: 
        emb = F.pad(emb, (0,1))
    return emb

# class EpsNet(nn.Module):
#     def __init__(self, in_channels=3, tdim=256, hidden=128):
#         super().__init__()
#         self.tb = nn.Sequential(nn.Linear(tdim, hidden),
#                                 nn.SiLU(), nn.Linear(hidden, hidden)) # B, H
#         self.up = nn.Sequential(
#             nn.Conv2d(in_channels, hidden, 3, padding=1),
#             nn.SiLU(),
#             nn.Conv2d(hidden, hidden, 3, padding=1),
#             nn.SiLU()
#         )
#         self.down = nn.Sequential(
#             nn.Conv2d(hidden, hidden, 3, padding=1),
#             nn.SiLU(),
#             nn.Conv2d(hidden, in_channels, 3, padding=1),
#         )


#     def forward(self, x, t):
#         tb = self.tb(timestep_embedding(t, 256))
#         tb = tb.view(tb.size(0), tb.size(1), 1, 1)
#         out = self.up(x) + tb
#         return self.down(out)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(tdim, out_ch))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.emb(temb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)

class EpsNet(nn.Module):
    def __init__(self, in_ch=1, base=128, tdim=256):
        super().__init__()
        self.tproj = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.emb_fn = lambda t: timestep_embedding(t, tdim)

        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
        self.down1 = ResBlock(base, base, tdim)
        self.ds1 = nn.Conv2d(base, base*2, 4, stride=2, padding=1)
        self.down2 = ResBlock(base*2, base*2, tdim)
        self.ds2 = nn.Conv2d(base*2, base*4, 4, stride=2, padding=1)

        self.mid = ResBlock(base*4, base*4, tdim)

        self.us1 = nn.ConvTranspose2d(base*4, base*2, 4, stride=2, padding=1)
        self.up1 = ResBlock(base*4, base*2, tdim)
        self.us2 = nn.ConvTranspose2d(base*2, base, 4, stride=2, padding=1)
        self.up2 = ResBlock(base*2, base, tdim)
        self.out = nn.Conv2d(base, in_ch, 3, padding=1)
        
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t):
        temb = self.tproj(self.emb_fn(t))

        x1 = self.in_conv(x)
        x2 = self.down1(x1, temb)
        x3 = self.ds1(x2)
        x4 = self.down2(x3, temb)
        x5 = self.ds2(x4)

        m  = self.mid(x5, temb)

        u1 = self.us1(m)
        u1 = torch.cat([u1, x4], 1)
        u1 = self.up1(u1, temb)
        u2 = self.us2(u1)
        u2 = torch.cat([u2, x2], 1)
        u2 = self.up2(u2, temb)
        return self.out(u2)
    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x*2 - 1),
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = EpsNet(1).to(device)

ema_decay = 0.999
ema = EpsNet(1).to(device)
ema.load_state_dict(model.state_dict())
for p in ema.parameters(): 
    p.requires_grad = False
# print(len(ema.parameters()))

def ema_update():
    with torch.no_grad():
        msd, esd = model.state_dict(), ema.state_dict()
        for k in msd:
            esd[k].mul_(ema_decay).add_(msd[k], alpha=1-ema_decay)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=3e-5)


alpha_bars = torch.cumprod(alphas, dim=0)

for epoch in range(num_epochs):
    loss = 0.0

    for inputs, labels in train_loader:
        x_0 = inputs.to(device) # B, 1, 28, 28
        B = inputs.size(0)
        t = torch.randint(0, max_steps, (B,), device=device)  # B
        
        alpha_bar_t = alpha_bars[t].view(B, 1, 1, 1)

        noise = torch.randn_like(x_0, device=device) # B, 1, 28, 28

        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1-alpha_bar_t) * noise
        predicted_noise = model(x_t, t)
        loss = F.mse_loss(predicted_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        ema_update()
    scheduler.step()

        
    print(loss.item())

    
# Inference
def calc_mean(x_t, t, noise):
    a_t = alphas[t].view(-1, 1, 1, 1)
    a_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
    betta_t = betas[t].view(-1, 1, 1, 1)
    return 1/torch.sqrt(a_t) * (x_t - betta_t/torch.sqrt(1-a_bar_t) * noise)

def calc_betta_bar(t):
    betta_t = betas[t].view(-1, 1, 1, 1)
    a_bar_prev = alpha_bars[t-1].view(-1, 1, 1, 1)
    a_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
    return betta_t * (1-a_bar_prev)/(1-a_bar_t)


def save_ckpt(path, model, ema=None, optimizer=None, scheduler=None, step=0, epoch=0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "ema": None if ema is None else ema.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
    }, path)

save_ckpt("check/mnist_ddpm.pt", model, ema=ema, optimizer=optimizer, scheduler=scheduler, epoch=epoch)

ema.eval()
with torch.no_grad():
    for _ in range(3):
        B = 1
        x_t = torch.randn(B, 1, 28, 28, device=device)
        for i in range(max_steps):
            t = torch.tensor([max_steps-i-1] * B, device=device)
            noise = ema(x_t, t)
            mu = calc_mean(x_t, t, noise)

            if (max_steps-1-i) > 0:
                z = torch.randn_like(x_t, device=device)

                x_t = mu + torch.sqrt(calc_betta_bar(t)) * z
            else:
                x_t = mu

        final = (x_t.clamp(-1, 1) + 1) / 2.0 
        one = final[0, 0].cpu().numpy()               # (28, 28)
        plt.figure()
        plt.axis('off')
        plt.imshow(one, cmap='gray', vmin=0, vmax=1)
        plt.show()


