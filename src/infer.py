import os
import time
import torch
from tqdm import tqdm
from models.coarse_reconstruction import CoarseSNN
from data_loader import get_loader
from utils.helpers import get_device
from metrics import compute_latency, compute_throughput, compute_memory_usage, compute_power_usage

# ----------------------------
# Config
# ----------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/u-caltech")
TIME_STEPS = 25
BATCH_SIZE = 8
METRICS_FILE = "inference_metrics.txt"

# ----------------------------
# Device
# ----------------------------
device = get_device("auto")
print(f"Running inference on {device}...")

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
# Inference + Metrics
# ----------------------------
latencies, total_samples = [], 0
torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    for spikes, _, _ in tqdm(test_loader, desc="Inference"):
        spikes = spikes.to(device).float()

        start = time.time()
        outputs = model(spikes)

        # Normalize output for fairness
        omin = outputs.amin(dim=(2,3), keepdim=True)
        omax = outputs.amax(dim=(2,3), keepdim=True)
        outputs = (outputs - omin) / (omax - omin + 1e-6)

        end = time.time()
        latencies.append(end - start)
        total_samples += spikes.size(0)

# ----------------------------
# Compute Metrics
# ----------------------------
avg_latency, std_latency = compute_latency(latencies)
throughput = compute_throughput(total_samples, sum(latencies))
memory_usage = compute_memory_usage()
power_usage = compute_power_usage()

metrics_text = (
    "\n===== INFERENCE METRICS =====\n"
    f"Avg Latency per batch: {avg_latency:.4f}s ± {std_latency:.4f}s\n"
    f"Throughput: {throughput:.2f} samples/sec\n"
    f"GPU Memory Usage: {memory_usage:.2f} MB\n"
    f"Average Power Draw: {power_usage:.2f} W\n"
    "==============================\n"
)

print(metrics_text)

# Save
with open(METRICS_FILE, "w") as f:
    f.write(metrics_text)

print(f"Inference metrics saved to {METRICS_FILE}")
