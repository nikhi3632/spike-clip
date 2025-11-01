import os
import torch
import matplotlib.pyplot as plt
from models.coarse_reconstruction import CoarseSNN
from data_loader import get_loader
from utils.helpers import get_device

# ----------------------------
# Config
# ----------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/u-caltech")
TIME_STEPS = 25
BATCH_SIZE = 1
SAVE_PATH = "test_visualization.png"

# ----------------------------
# Device
# ----------------------------
device = get_device("auto")
print(f"Using device: {device}")

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

test_loader = get_loader(DATA_DIR, LABELS, split="test", batch_size=BATCH_SIZE)

# ----------------------------
# Model
# ----------------------------
model = CoarseSNN(time_steps=TIME_STEPS).to(device)
checkpoint_path = "checkpoints/coarse_snn.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# ----------------------------
# Visualization
# ----------------------------
spikes, labels, _ = next(iter(test_loader))
spikes = spikes.to(device).float()

with torch.no_grad():
    outputs = model(spikes)
    omin = outputs.amin(dim=(2,3), keepdim=True)
    omax = outputs.amax(dim=(2,3), keepdim=True)
    outputs = (outputs - omin) / (omax - omin + 1e-6)

# Temporal sum (input preview)
input_frame = spikes.sum(dim=1).squeeze().cpu().numpy()

# Output (normalized)
output_img = outputs.squeeze().permute(1,2,0).cpu().numpy()

# ----------------------------
# Plot and save
# ----------------------------
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title(f"Input (Event Frame Sum)\nLabel: {labels[0]}")
plt.imshow(input_frame, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Reconstructed Output")
plt.imshow(output_img)
plt.axis("off")

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300)
plt.show()

print(f"Visualization saved to {SAVE_PATH}")
