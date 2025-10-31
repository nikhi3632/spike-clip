# train.py (updated with logging)
import os
import time
import torch
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
EPOCHS = 7
LEARNING_RATE = 1e-2
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
# Dataset & Loaders
# ----------------------------
LABELS = ['accordion','airplanes','anchor','ant','barrel','bass','beaver','binocular','bonsai','brain','brontosaurus','buddha','butterfly','camera','cannon','car','ceilingfan','cellphone','chair','chandelier','cougarbody','cougarface','crab','crayfish','crocodile','crocodilehead','cup','dalmatian','dollarbill','dolphin','dragonfly','electricguitar','elephant','emu','euphonium','ewer','faces','ferry','flamingo','flamingohead','garfield','gerenuk','gramophone','grandpiano','hawksbill','headphone','hedgehog','helicopter','ibis','inlineskate','joshuatree','kangaroo','ketch','lamp','laptop','Leopards','llama','lobster','lotus','mandolin','mayfly','menorah','metronome','minaret','Motorbikes','nautilus','octopus','okapi','pagoda','panda','pigeon','pizza','platypus','pyramid','revolver','rhino','rooster','saxophone','schooner','scissors','scorpion','seahorse','snoopy','soccerball','stapler','starfish','stegosaurus','stopsign','strawberry','sunflower','tick','trilobite','umbrella','watch','waterlilly','wheelchair','wildcat','windsorchair','wrench','yinyang','background']
train_loader = get_loader(DATA_DIR, LABELS, split="train", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# ----------------------------
# Model, Loss, Optimizer
# ----------------------------
model = CoarseSNN(time_steps=TIME_STEPS).to(device)
criterion = ReconstructionLoss(loss_type="l1")
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler(enabled=(device=="cuda"))

# ----------------------------
# Create log file and write config
# ----------------------------
with open(LOG_FILE, "w") as f:
    f.write("==== TRAINING LOG ====\n")
    f.write(f"Timestamp: {time.ctime()}\n")
    f.write(f"Device: {device}\n")
    f.write(f"Dataset: {DATA_DIR}\n")
    f.write(f"Number of classes: {len(LABELS)}\n")
    f.write(f"Labels: {LABELS}\n")
    f.write(f"EPOCHS: {EPOCHS}, LEARNING_RATE: {LEARNING_RATE}, TIME_STEPS: {TIME_STEPS}\n")
    f.write(f"BATCH_SIZE: {BATCH_SIZE}, NUM_WORKERS: {NUM_WORKERS}, PIN_MEMORY: {PIN_MEMORY}\n\n")
    f.write("Epoch\tAvg Loss\tTime (s)\n")

# ----------------------------
# Training Loop
# ----------------------------
for epoch in range(EPOCHS):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    
    for spikes, _, _ in pbar:
        spikes = spikes.to(device)
        target = torch.rand(spikes.shape[0], 3, 224, 224).to(device)  # placeholder Ground Truth
        
        optimizer.zero_grad()
        with autocast(device_type='cuda', enabled=(device=="cuda")):
            outputs = model(spikes)
            loss = criterion(outputs, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        pbar.set_postfix({"Loss": running_loss / (pbar.n + 1)})
    
    epoch_time = time.time() - epoch_start
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
    
    # log to file
    with open(LOG_FILE, "a") as f:
        f.write(f"{epoch+1}\t{avg_loss:.4f}\t{epoch_time:.2f}\n")

# ----------------------------
# Save Model
# ----------------------------
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/coarse_snn.pth")
print("Stage 1 SNN model saved to checkpoints/coarse_snn.pth")

with open(LOG_FILE, "a") as f:
    f.write("Training completed.\n")
