# test.py — GPU inference test + visualization
import torch
import os
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

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

LABELS = [
    'accordion','airplanes','anchor','ant','barrel','bass','beaver','binocular','bonsai','brain','brontosaurus',
    'buddha','butterfly','camera','cannon','car','ceilingfan','cellphone','chair','chandelier','cougarbody',
    'cougarface','crab','crayfish','crocodile','crocodilehead','cup','dalmatian','dollarbill','dolphin',
    'dragonfly','electricguitar','elephant','emu','euphonium','ewer','faces','ferry','flamingo','flamingohead',
    'garfield','gerenuk','gramophone','grandpiano','hawksbill','headphone','hedgehog','helicopter','ibis',
    'inlineskate','joshuatree','kangaroo','ketch','lamp','laptop','Leopards','llama','lobster','lotus',
    'mandolin','mayfly','menorah','metronome','minaret','Motorbikes','nautilus','octopus','okapi','pagoda',
    'panda','pigeon','pizza','platypus','pyramid','revolver','rhino','rooster','saxophone','schooner',
    'scissors','scorpion','seahorse','snoopy','soccerball','stapler','starfish','stegosaurus','stopsign',
    'strawberry','sunflower','tick','trilobite','umbrella','watch','waterlilly','wheelchair','wildcat',
    'windsorchair','wrench','yinyang','background'
]

# ----------------------------
# Device Setup
# ----------------------------
device = get_device("auto")
print(f"Using device: {device}")

# ----------------------------
# Load Dataset
# ----------------------------
test_loader = get_loader(DATA_DIR, LABELS, split="test", batch_size=BATCH_SIZE)
spikes, images, labels = next(iter(test_loader))

spikes = spikes.to(device)
print(f"Loaded real sample from dataset with shape: {spikes.shape}")

# ----------------------------
# Load Model
# ----------------------------
model = CoarseSNN(time_steps=TIME_STEPS).to(device)

if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print(f"Loaded trained weights from {CHECKPOINT_PATH}")
else:
    print("⚠️ No checkpoint found. Using randomly initialized weights.")

model.eval()

# ----------------------------
# Forward Pass (Inference Test)
# ----------------------------
with torch.no_grad():
    output = model(spikes)

# ----------------------------
# Convert for Visualization
# ----------------------------
# input visualization (sum over time steps)
input_image = spikes[0].sum(dim=0)  # [224,224]
input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())

# output visualization
output_image = output[0].detach().cpu()
output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Input (Event Frame Sum)")
plt.imshow(input_image.cpu(), cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Reconstructed Output")
plt.imshow(to_pil_image(output_image))
plt.axis("off")

plt.tight_layout()
plt.savefig("test_visualization.png")
plt.show()

print("Saved visualization to test_visualization.png")

# ----------------------------
# Output Stats
# ----------------------------
print("\n===== TEST OUTPUT (REAL SAMPLE) =====")
print(f"Input shape:   {tuple(spikes.shape)}")
print(f"Output shape:  {tuple(output.shape)}")
print(f"Output device: {output.device}")
print(f"Output dtype:  {output.dtype}")
print(f"Output mean:   {output.mean().item():.4f}")
print(f"Output std:    {output.std().item():.4f}")
print("========================")
