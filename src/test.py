# test.py
import os
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T

from models.coarse_reconstruction import CoarseSNN
from data_loader import get_loader
from utils.helpers import get_device

# ----------------------------
# Config
# ----------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/u-caltech")
TIME_STEPS = 25
BATCH_SIZE = 1
CHECKPOINT_PATH = "checkpoints/coarse_snn.pth"

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

test_loader = get_loader(DATA_DIR, LABELS, split="test", batch_size=BATCH_SIZE, shuffle=True)
spikes, _, label_idx = next(iter(test_loader))
label_name = LABELS[label_idx[0]]

spikes = spikes.to(device)

# ----------------------------
# Model
# ----------------------------
model = CoarseSNN(time_steps=TIME_STEPS).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

# ----------------------------
# Inference
# ----------------------------
with torch.no_grad():
    output = model(spikes)

# ----------------------------
# Visualization Prep
# ----------------------------
# Sum spikes across time for visualization
input_sum = spikes.sum(dim=1).squeeze().cpu()
input_sum = (input_sum - input_sum.min()) / (input_sum.max() - input_sum.min() + 1e-8)

# Output postprocessing & contrast enhancement
output_img = output[0].detach().cpu()
output_img = torch.sigmoid(output_img)  # ensure values [0,1]
output_img = torch.clamp(output_img * 5, 0, 1)  # contrast boost

# Convert to numpy
input_np = input_sum.numpy()
output_np = output_img.permute(1, 2, 0).numpy()

# ----------------------------
# Plot
# ----------------------------
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(input_np, cmap="gray")
axs[0].set_title(f"Input (Event Frame Sum)\nLabel: {label_name}")
axs[0].axis("off")

axs[1].imshow(output_np)
axs[1].set_title("Reconstructed Output")
axs[1].axis("off")

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/test_visualization.png", dpi=300)
plt.show()

print("Saved visualization to outputs/test_visualization.png")
