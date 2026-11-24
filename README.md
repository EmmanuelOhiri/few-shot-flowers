# Few-shot flower classification

Code and results for the CUDO Compute article [Few-shot learning: everything you need to know](https://www.cudocompute.com/blog/few-shot-learning-everything-you-need-to-know). The notebook compares four ways to classify flower species that no model in it was trained on, given between one and twenty labeled photos of each species.

## What the notebook measures

[Oxford Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) holds 8,189 photos of 102 flower species. A seeded random split puts 62 species in a training set and holds the other 40 back for testing, so every test species is new to every model. Each score is the mean accuracy over 600 random episodes drawn from the 40 test species, with 15 query photos per species and a 95% confidence interval. Every method is scored on exactly the same episodes.

| Method | How it classifies a query photo |
|---|---|
| Untrained network + prototypes | A ResNet-18 with random weights embeds every photo; the query takes the label of the nearest class prototype, the mean embedding of that class's support photos |
| ImageNet network + prototypes | The same, with ImageNet-pretrained weights, frozen |
| ImageNet network + logistic regression | A logistic regression fitted on the support photos' frozen, length-normalized embeddings |
| ImageNet network + episodic training | The pretrained network trained further as a prototypical network on 2,000 five-way, five-shot episodes from the 62 training species, then classified by prototypes |

## Results

Accuracy with five species per episode, by the number of labeled photos per species (95% intervals are within ±1 point):

| Method | 1 | 2 | 5 | 10 | 20 |
|---|---|---|---|---|---|
| Untrained network + prototypes | 36.2% | 40.8% | 45.6% | 48.3% | 49.5% |
| ImageNet network + prototypes | 76.0% | 89.2% | 95.5% | 96.8% | 97.6% |
| ImageNet network + logistic regression | **83.7%** | 91.7% | 96.1% | 97.3% | 98.1% |
| ImageNet network + episodic training | 81.0% | **93.9%** | **97.9%** | **98.7%** | **99.2%** |

Accuracy with more species per episode:

| Method | 10 species, 1-shot | 20 species, 1-shot | 10 species, 5-shot | 20 species, 5-shot |
|---|---|---|---|---|
| Chance | 10.0% | 5.0% | 10.0% | 5.0% |
| Untrained network + prototypes | 24.5% | 16.6% | 31.9% | 22.6% |
| ImageNet network + prototypes | 68.0% | 60.1% | 92.0% | 87.8% |
| ImageNet network + logistic regression | **76.9%** | **68.5%** | 92.2% | 86.9% |
| ImageNet network + episodic training | 74.2% | 65.6% | **96.3%** | **93.9%** |

Episodic training, scored on the 40 test species every 250 training episodes. The loss columns cover the 250 training episodes before each checkpoint. These checkpoints were recorded for reporting only; every result above uses the model at episode 2,000.

| Episode | 5 species, 1-shot | 5 species, 5-shot | Mean loss | Median loss | Share of episodes with loss below 0.001 |
|---|---|---|---|---|---|
| 0 | 76.0% | 95.5% | | | |
| 250 | 77.9% | 97.2% | 1.223 | 0.912 | 16% |
| 500 | 80.5% | 97.5% | 0.864 | 0.526 | 25% |
| 750 | 81.5% | 97.7% | 0.574 | 0.241 | 33% |
| 1,000 | 81.9% | 97.8% | 0.400 | 0.130 | 35% |
| 1,250 | 81.7% | 97.8% | 0.415 | 0.102 | 36% |
| 1,500 | 82.2% | 97.9% | 0.290 | 0.008 | 48% |
| 1,750 | 81.2% | 97.8% | 0.243 | 0.001 | 50% |
| 2,000 | 81.0% | 97.9% | 0.219 | 0.0001 | 56% |

These numbers come from one dataset, one backbone and one random split of species. They show the size of each effect on this data, not a general ranking of the methods.

## Files

| File | Contents |
|---|---|
| `few_shot_flowers.ipynb` | The notebook as run, with its saved output. The first part runs the four-method comparison; the "Extra measurements" section adds the 1 to 20 photo grid, the 10- and 20-species settings and the training checkpoints. |
| `few_shot_flowers.py` | The first part of the notebook as a standalone script. It does not include the extra measurements. |
| `results/extra_results.json`, `results/extra_results.csv` | Every number from the "Extra measurements" section, including a 20-episode quick check that is not a result. |
| `requirements.txt` | The Python packages both files need. |

## Running it

### In a notebook

1. Open `few_shot_flowers.ipynb` in Jupyter or any notebook environment with a GPU.
2. Install the packages in `requirements.txt` if the environment does not have them.
3. Run the cells in order. Do not skip the definition cells at the top; the later sections reuse them.

On a single NVIDIA T4 GPU, the first part took 4,515 seconds, about 75 minutes, most of it in the 2,000 training episodes. The "Extra measurements" section took 13,595 seconds from its first cell to its last, including any pauses between cells. It retrains the episodic model from scratch with checkpoints, and the logistic regression runs for 20 species with five photos each take about 17 minutes on their own.

### On your own machine

```bash
pip install -r requirements.txt
python few_shot_flowers.py
```

The script uses the GPU when PyTorch can find one and falls back to the CPU, where training is much slower. Add `--dry-run` to check that everything runs, in under a minute on random images, without downloading anything. Add `--workers 4` (or any number of CPU cores) to load images in parallel; see the note on reproducibility below.

## Reproducibility

- **Seeds:** The species split uses seed 0, the test episodes seed 1 and the training episodes seed 2, so every method sees the same episodes on every run.
- **Data loading:** The notebook loads images in the main process (`workers = 0`), because worker processes forked inside a notebook can hang. With more workers, the random crops and flips used during training come from a different random stream, so the episodic-training results can differ slightly from the tables above. The frozen-embedding results do not change.
- **Hardware and versions:** The recorded runs used a notebook on a single NVIDIA T4 GPU, with Python 3.13, torch 2.14.1+cu130 and torchvision 0.29.1+cu130. The second run, in the "Extra measurements" section, reproduced the first run's numbers exactly on that setup. Other GPUs or library versions can change the episodic-training results slightly.

## Method notes

- Prototypes compare unnormalized 512-dimensional embeddings by squared Euclidean distance, following the [prototypical networks paper](https://arxiv.org/abs/1703.05175). The logistic regression uses scikit-learn's defaults on length-normalized embeddings, so the comparison between the two does not separate the effect of the classifier from the effect of normalization.
- Training uses Adam with a learning rate of 1e-5 and five-way, five-shot episodes with 10 queries per species, with random resized crops and horizontal flips. Test photos are resized and center-cropped only.
- The pretrained weights are torchvision's `ResNet18_Weights.IMAGENET1K_V1`. ImageNet includes a few flower categories, such as daisy, so the pretrained network has seen some flowers before, but none of its 1,000 categories is trained to separate these 40 species from one another.

## Data and license

Oxford Flowers-102 was created by Maria-Elena Nilsback and Andrew Zisserman at the Visual Geometry Group, University of Oxford: M.-E. Nilsback and A. Zisserman, "Automated flower classification over a large number of classes," Indian Conference on Computer Vision, Graphics and Image Processing, 2008. torchvision downloads it on the first run; this repository does not redistribute it.

The code in this repository is released under the [MIT License](LICENSE). The license covers the code only, not the Flowers-102 images.
