"""Few-shot image classification on flower species, scored over many random episodes.

Compares four ways to classify 5-way 1-shot and 5-way 5-shot episodes drawn from
40 Oxford Flowers-102 species that no model below is trained on:
  1. an untrained ResNet-18 (random weights) as the embedding, classified by prototypes
  2. an ImageNet-pretrained ResNet-18, frozen, classified by prototypes
  3. the same frozen embedding with a logistic regression fitted on the support set
  4. the pretrained ResNet-18 after episodic prototypical training on the other 62 species

Needs: torch, torchvision, scikit-learn, scipy (torchvision downloads Flowers-102).
Run:   python few_shot_flowers.py
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="./data")
parser.add_argument("--episodes", type=int, default=600, help="test episodes per setting")
parser.add_argument("--train-episodes", type=int, default=2000)
parser.add_argument("--workers", type=int, default=0,
                    help="DataLoader worker processes; 0 loads in the main process (workers forked in a notebook can hang)")
parser.add_argument("--dry-run", action="store_true", help="random images, no downloads")
args, _ = parser.parse_known_args()  # also works when pasted into a notebook cell

SEED = 0
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
start = time.time()

# ---------------------------------------------------------------- data
mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
eval_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                              transforms.ToTensor(), transforms.Normalize(mean, std)])
train_tf = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(), transforms.Normalize(mean, std)])

def flowers(transform):
    # Pool the official train, val and test splits: the episodes below need every species
    splits = [datasets.Flowers102(args.data, split=s, transform=transform, download=True)
              for s in ("train", "val", "test")]
    labels = np.concatenate([np.asarray(d._labels) for d in splits])
    return ConcatDataset(splits), labels

if args.dry_run:
    g = torch.Generator().manual_seed(SEED)
    labels = np.repeat(np.arange(102), 25)
    images = torch.randn(len(labels), 3, 64, 64, generator=g)
    eval_set = train_set = TensorDataset(images, torch.as_tensor(labels))
    args.episodes, args.train_episodes = 20, 5
else:
    eval_set, labels = flowers(eval_tf)
    train_set, _ = flowers(train_tf)

rng = np.random.default_rng(SEED)
species = rng.permutation(102)
base_classes, novel_classes = np.sort(species[:62]), np.sort(species[62:])
by_class = {c: np.flatnonzero(labels == c) for c in range(102)}
print(f"device: {device}  images: {len(labels)}  "
      f"base species: {len(base_classes)}  novel species: {len(novel_classes)}")

def sample_episode(rng, classes, n_way, k_shot, n_query):
    """Pick n_way classes; return support and query image indices with labels 0..n_way-1."""
    chosen = rng.choice(classes, n_way, replace=False)
    s_idx, s_lab, q_idx, q_lab = [], [], [], []
    for j, c in enumerate(chosen):
        picks = rng.choice(by_class[c], k_shot + n_query, replace=False)
        s_idx += picks[:k_shot].tolist(); s_lab += [j] * k_shot
        q_idx += picks[k_shot:].tolist(); q_lab += [j] * n_query
    return s_idx, torch.tensor(s_lab), q_idx, torch.tensor(q_lab)

# ---------------------------------------------------------------- model
def backbone(pretrained):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Identity()  # keep the 512-dimensional embedding, drop ImageNet's classifier
    return model.to(device)

@torch.no_grad()
def embed(model, indices):
    model.eval()
    loader = DataLoader(Subset(eval_set, indices), batch_size=256, num_workers=args.workers)
    return torch.cat([model(x.to(device)).cpu() for x, _ in loader])

def prototype_logits(f_support, y_support, f_query, n_way):
    # Snell et al. (2017): each prototype is the mean embedding of its class's support images;
    # a query scores each class by negative squared Euclidean distance to its prototype
    protos = torch.stack([f_support[y_support == j].mean(0) for j in range(n_way)])
    return -torch.cdist(f_query, protos) ** 2

# ---------------------------------------------------------------- evaluation
novel_idx = np.concatenate([by_class[c] for c in novel_classes])
row_of = {i: r for r, i in enumerate(novel_idx)}  # image index -> row in the feature table

def evaluate(features, classifier, n_way, k_shot, n_query=15):
    rng = np.random.default_rng(SEED + 1)  # same episodes for every method
    accs = []
    for _ in range(args.episodes):
        s, ys, q, yq = sample_episode(rng, novel_classes, n_way, k_shot, n_query)
        fs, fq = features[[row_of[i] for i in s]], features[[row_of[i] for i in q]]
        if classifier == "prototypes":
            pred = prototype_logits(fs, ys, fq, n_way).argmax(1)
        else:
            fs, fq = F.normalize(fs, dim=1), F.normalize(fq, dim=1)
            clf = LogisticRegression(max_iter=1000).fit(fs.numpy(), ys.numpy())
            pred = torch.as_tensor(clf.predict(fq.numpy()))
        accs.append((pred == yq).float().mean().item())
    accs = np.array(accs) * 100
    return accs.mean(), 1.96 * accs.std() / np.sqrt(len(accs))  # mean and 95% interval

results = {}
def run(name, model, classifier):
    t0 = time.time()
    features = embed(model, novel_idx)
    t1 = time.time()
    results[name] = [evaluate(features, classifier, 5, k) for k in (1, 5)]
    print(f"{name}: embedding {t1 - t0:.0f} s, episodes {time.time() - t1:.0f} s", flush=True)

run("Untrained ResNet-18 + prototypes", backbone(False), "prototypes")
pretrained = backbone(True if not args.dry_run else False)
run("ImageNet ResNet-18 + prototypes", pretrained, "prototypes")
run("ImageNet ResNet-18 + logistic regression", pretrained, "logistic")

# ---------------------------------------------------------------- episodic training on base species
class EpisodeSampler:
    """Yields one training episode (5-way, 5-shot, 10 queries per class) per batch."""
    def __init__(self, n_episodes, n_way=5, k_shot=5, n_query=10):
        self.n, self.n_way, self.k, self.q = n_episodes, n_way, k_shot, n_query
        self.rng = np.random.default_rng(SEED + 2)
    def __len__(self):
        return self.n
    def __iter__(self):
        for _ in range(self.n):
            s, _, q, _ = sample_episode(self.rng, base_classes, self.n_way, self.k, self.q)
            yield s + q

sampler = EpisodeSampler(args.train_episodes)
loader = DataLoader(train_set, batch_sampler=sampler, num_workers=args.workers)
model = backbone(True if not args.dry_run else False)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
n_way, k_shot, n_query = sampler.n_way, sampler.k, sampler.q
y_support = torch.arange(n_way).repeat_interleave(k_shot).to(device)
y_query = torch.arange(n_way).repeat_interleave(n_query).to(device)

model.train()
for step, (images, _) in enumerate(loader, 1):
    # sample_episode orders images class by class, so these label tensors line up with the batch
    feats = model(images.to(device))
    f_support, f_query = feats[: n_way * k_shot], feats[n_way * k_shot:]
    loss = F.cross_entropy(prototype_logits(f_support, y_support, f_query, n_way), y_query)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 250 == 0 or step == len(sampler):
        print(f"  training episode {step}/{len(sampler)}  loss {loss.item():.3f}")

run("ImageNet ResNet-18 + episodic training + prototypes", model, "prototypes")

# ---------------------------------------------------------------- report
print(f"\nAccuracy on {len(novel_classes)} unseen species, mean of {args.episodes} episodes "
      f"(95% interval), 15 queries per class")
print(f"{'method':<54}{'5-way 1-shot':>16}{'5-way 5-shot':>16}")
for name, ((m1, c1), (m5, c5)) in results.items():
    print(f"{name:<54}{m1:>9.1f} ± {c1:<4.1f}{m5:>9.1f} ± {c5:<4.1f}")
gpu = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
print(f"\n{gpu}, torch {torch.__version__}, torchvision "
      f"{__import__('torchvision').__version__}, {time.time() - start:.0f} s")
