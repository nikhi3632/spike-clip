import os
import time
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from data_loader import get_loader
from utils.helpers import get_device
from loss import ReconstructionLoss
from models.coarse_reconstruction import CoarseSNN

# ----------------------------
# Config
# ----------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/u-caltech")
EPOCHS = 10
LEARNING_RATE = 1e-3
TIME_STEPS = 25
BATCH_SIZE = 8
NUM_WORKERS = 8
PIN_MEMORY = True
LOG_FILE = "train_log.txt"

# ----------------------------
# Device
# ----------------------------
device = get_device("auto")
print(f"Using device: {device}")

if device != "cuda":
    BATCH_SIZE = 2
    NUM_WORKERS = 0
    PIN_MEMORY = False

# ----------------------------
# Dataset
# ----------------------------
LABELS = ['accordion','airplanes','anchor','ant','barrel','bass','beaver','binocular','bonsai','brain','brontosaurus',
          'buddha','butterfly','camera','cannon','car','ceilingfan','cellphone','chair','chandelier','cougarbody',
          'cougarface','crab','crayfish','crocodile','crocodilehead','cup','dalmatian','dollarbill','dolphin',
          'dragonfly','electricguitar','elephant','emu','euphonium','ewer','faces','ferry','flamingo','flamingohead',
          'garfield','gerenuk','gramophone','grandpiano','hawksbill','headphone','hedgehog','helicopter','ibis',
          'inlineskate','joshuatree','kangaroo','ketch','lamp','laptop','Leopards','llama','lobster','lotus',
          'mandolin','mayfly','menorah','metronome','minaret','Motorbikes','nautilus','octopus','okapi','pagoda',
          'panda','pigeon','pizza','platypus','pyramid','revolver','rhino','rooster','saxophone','schooner',
          'scissors','scorpion','seahorse','snoopy','soccerball','stapler','starfish','stegosaurus','stopsign',
          'strawberry','sunflower','tick','trilobite','umbrella','watch','waterlilly','wheelchair','wildcat',
          'windsorchair','wrench','yinyang','background']

train_loader = get_loader(DATA_DIR, LABELS, split="train", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# ----------------------------
# Helper functions
# ----------------------------
def make_pseudo_target(spikes):
    """Temporal mean + normalization"""
    tmean = spikes.mean(dim=1, keepdim=True)
    tmin = tmean.amin(dim=(2,3), keepdim=True)
    tmax = tmean.amax(dim=(2,3), keepdim=True)
    target = (tmean - tmin) / (tmax - tmin + 1e-6)
    target = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    return target.repeat(1, 3, 1, 1)

def sobel_edges(x):
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    ky = kx.transpose(2,3)
    gx = F.conv2d(x, kx, padding=1, groups=x.shape[1])
    gy = F.conv2d(x, ky, padding=1, groups=x.shape[1])
    return torch.sqrt(gx*gx + gy*gy + 1e-6)

def tv_loss(x):
    return (x[:,:,:, :-1] - x[:,:,:,1:]).abs().mean() + (x[:,:,:-1,:] - x[:,:,1:,:]).abs().mean()

# ----------------------------
# Model, Loss, Optimizer
# ----------------------------
model = CoarseSNN(time_steps=TIME_STEPS).to(device)
criterion = ReconstructionLoss(loss_type="l1")
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler(enabled=(device == "cuda"))

# ----------------------------
# Logging setup
# ----------------------------
with open(LOG_FILE, "w") as f:
    f.write("==== TRAINING LOG ====\n")
    f.write(f"Timestamp: {time.ctime()}\nDevice: {device}\nDataset: {DATA_DIR}\n")
    f.write(f"EPOCHS={EPOCHS}, LR={LEARNING_RATE}, TIME_STEPS={TIME_STEPS}\n")
    f.write(f"BATCH_SIZE={BATCH_SIZE}, WORKERS={NUM_WORKERS}\n\n")
    f.write("Epoch\tAvgLoss\tTime(s)\tBest?\n")

# ----------------------------
# Training Loop (with best model saving)
# ----------------------------
best_loss = float("inf")
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(EPOCHS):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

    for i, (spikes, _, _) in enumerate(pbar):
        spikes = spikes.to(device).float()
        target = make_pseudo_target(spikes)

        optimizer.zero_grad()
        with autocast(device_type='cuda', enabled=(device == "cuda")):
            outputs = model(spikes)
            omin = outputs.amin(dim=(2,3), keepdim=True)
            omax = outputs.amax(dim=(2,3), keepdim=True)
            outputs_n = (outputs - omin) / (omax - omin + 1e-6)

            l_rec = F.l1_loss(outputs_n, target)
            edges_out = sobel_edges(outputs_n.mean(1, keepdim=True))
            edges_tgt = sobel_edges(target.mean(1, keepdim=True))
            l_edge = F.l1_loss(edges_out, edges_tgt)
            l_tv = tv_loss(outputs_n)

            loss = l_rec + 0.1*l_edge + 0.001*l_tv

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix({"Loss": running_loss / (pbar.n + 1)})

        if i == 0 and epoch == 0:
            print("Target stats:", float(target.mean()), float(target.min()), float(target.max()))
            print("Output stats:", float(outputs_n.mean()), float(outputs_n.min()), float(outputs_n.max()))

    # Epoch summary
    epoch_time = time.time() - epoch_start
    avg_loss = running_loss / len(train_loader)
    is_best = avg_loss < best_loss

    print(f"Epoch [{epoch+1}/{EPOCHS}] Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s {'<- Best so far' if is_best else ''}")

    # Save best model
    if is_best:
        best_loss = avg_loss
        best_path = "checkpoints/best_coarse_snn.pth"
        torch.save(model.state_dict(), best_path)

    # Log to file
    with open(LOG_FILE, "a") as f:
        f.write(f"{epoch+1}\t{avg_loss:.4f}\t{epoch_time:.2f}\t{'YES' if is_best else 'NO'}\n")

# Final save
final_path = "checkpoints/coarse_snn_final.pth"
torch.save(model.state_dict(), final_path)
print(f"✅ Training complete. Best model: {best_path} | Final model: {final_path}")

with open(LOG_FILE, "a") as f:
    f.write(f"Training completed.\nBest loss={best_loss:.4f}\n")
