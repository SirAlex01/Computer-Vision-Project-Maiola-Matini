# Video Frame Prediction via Attention-Based Spatio-Temporal Learning

**Authors:** Alessio Maiola, Alessandro Matini  
**Course:** Computer Vision — Sapienza University of Rome

---

## Abstract

Video frame prediction is a challenging task that requires a model to generate future frames given a sequence of past observations. While many approaches leverage pretext tasks such as Depth Estimation or Optical Flow to improve prediction quality, an alternative line of research operates directly on raw RGB or grayscale video frames, avoiding additional data preprocessing. Our work follows this latter direction.

We studied three recent papers proposing attention-based mechanisms for spatio-temporal video understanding, and designed a unified architecture that combines their core ideas:

1. **Temporal Attention Unit (TAU)** — capturing intra-frame (spatial) and inter-frame (temporal) dependencies through static and dynamic attention branches ([Tan et al., 2022](https://arxiv.org/abs/2206.12126)).
2. **Convolutional Block Attention Module (CBAM)** — enhancing feature representations via sequential channel and spatial attention ([Woo et al., 2018](https://arxiv.org/abs/1807.06521v2)).
3. **Receptive Field Attention (RFA)** — mitigating the parameter-sharing problem of standard convolutions by leveraging the full receptive field when computing attention with large filters ([Zhang et al., 2023](https://arxiv.org/abs/2304.03198)).

We experimented with multiple configurations of these components — varying their combination, ordering, and positioning — within a U-Net-style Encoder–Attention–Decoder architecture. All experiments were conducted on Google Colab due to the unavailability of dedicated GPU resources, which constrained training duration and hyperparameter exploration.

---

## Problem Statement and Approach

### Studying the Literature

We began by thoroughly reviewing the three foundational papers listed above. Each addresses a different aspect of the video prediction problem:

- **TAU** decomposes attention into a *static* branch (depthwise and dilated depthwise convolutions followed by a pointwise projection) that captures spatial structure within individual frames, and a *dynamic* branch (squeeze-and-excitation over temporally pooled features) that models how frame content evolves over time. Their combination is element-wise: `output = dynamic_attention × static_attention × features`.

- **CBAM** applies channel attention (via parallel average-pooling and max-pooling paths through a shared MLP) followed by spatial attention (a learned convolution over concatenated pooled maps). This sequential refinement allows the network to focus on *what* (channel) and *where* (spatial) the most informative features are.

- **RFA** addresses the parameter-sharing limitation of standard convolutions: a single convolutional filter applies the same weights everywhere regardless of local context. RFA first expands the feature map by a factor of $K^2$ (where $K$ is the kernel size) using a depthwise convolution, rearranges the result to explicitly represent each receptive field, and then applies attention at this expanded resolution before rescaling back. This allows the network to make spatially-varying use of its receptive fields.

After studying each mechanism in isolation, we designed an architecture that integrates all three ideas into a single pipeline, enabling us to evaluate their individual and combined contributions.

### Architecture

Our model, **STPLConvNet** (Spatio-Temporal Predictive Learning Convolutional Network), follows an Encoder–Attention Layer–Decoder structure:

```
Input (B×T×C×H×W) → Encoder → Attention Layer → Decoder → Output (B×T×C×H×W)
```

- **Encoder**: A stack of 4 convolutional blocks with GroupNorm and SiLU activations. Selective downsampling (stride-2 convolutions) progressively compresses spatial dimensions while collecting residual feature maps at each scale.
- **Attention Layer**: A configurable stack of 8 attention blocks (with stochastic depth regularization via DropPath) that operate on the encoded hidden state. The type and arrangement of attention blocks define the model variant (see below).
- **Decoder**: Mirrors the encoder using PixelShuffle-based upsampling and skip connections from the encoder residuals, culminating in a 1×1 convolution that projects back to the original channel count.

### Model Configurations

We experimented with three model types, each combining the attention mechanisms differently:

| Model Type | Attention Position | Description |
|:---:|:---:|:---|
| **0** | 0 | TAU only — temporal attention blocks followed by a Convolutional FFN |
| **1** | 0 | TAU → CBAM — temporal attention first, then spatial/channel refinement via CBAM, followed by ConvFFN |
| **2** | 2 | RFTAU ↔ RFCBAM interleaved — alternating receptive-field-enhanced versions of TAU and CBAM |

Additional orderings (CBAM → TAU, reversed RF variants) were also explored but the configurations above were selected for final evaluation.

### Loss Function

We employed a custom composite loss:

$$\mathcal{L} = \text{MSE}(\hat{Y}, Y) + \alpha \cdot \text{KL}\!\left(\text{softmax}\!\left(\frac{\Delta\hat{Y}}{\tau}\right) \;\Big\|\; \text{softmax}\!\left(\frac{\Delta Y}{\tau}\right)\right)$$

where $\Delta Y_t = Y_{t+1} - Y_t$ represents temporal frame differences, $\tau = 0.1$ is a temperature parameter, and $\alpha = 100$. The MSE term enforces per-pixel reconstruction accuracy, while the KL divergence over softmax-normalized frame differences encourages the model to capture correct *motion dynamics* between consecutive predicted frames, not just static appearance.

---

## Experimental Setup

### Datasets

| | **UCF101** | **MovingMNIST** |
|:---|:---:|:---:|
| **Type** | Real-world RGB | Synthetic grayscale |
| **Content** | 101 human action categories | Two moving MNIST digits |
| **Channels** | 3 | 1 |
| **Resolution** | 240 × 320 | 64 × 64 |
| **Frames per clip** | 10 (5 input → 5 predicted) | 20 (10 input → 10 predicted) |

### Training Details

- **Optimizer**: AdamW (lr = 1e-3, weight decay = 5e-4)
- **Scheduler**: OneCycleLR (max_lr = 1e-3)
- **Batch size**: 8
- **Epochs**: 100 (MovingMNIST), 50 target / 8 completed (UCF101)
- **Reproducibility**: Fixed seed (42) across all random generators, deterministic cuDNN

### Computational Constraints

All experiments were run on **Google Colab**, which imposed significant limitations. A single epoch on UCF101 required approximately 1 hour of GPU time (~3,800s for model type 0, up to ~6,600s for model type 2 with receptive field convolutions), making it impractical to complete the full 50-epoch schedule. UCF101 models were trained for only 8 epochs. MovingMNIST, being smaller and lower-resolution, allowed us to complete the full 100 epochs (~3 minutes per epoch for model type 0).

### Reproducibility Infrastructure

To cope with Colab's session time limits and ensure no training progress was lost, we implemented a checkpoint-based experiment management system:

- **Automatic checkpointing**: model weights, optimizer state, scheduler state, and current epoch are saved after every epoch.
- **Seamless resume**: training automatically resumes from the latest checkpoint when a Colab session is restarted.
- **Best model tracking**: the model with the lowest validation loss is saved separately for evaluation.
- **Structured logging**: per-epoch training and validation losses, along with elapsed time, are written to persistent log files.

---

## Results

### Quantitative Results

Best validation loss achieved by each configuration:

| Dataset | Model 0/0 (TAU) | Model 1/0 (TAU+CBAM) | Model 2/2 (RFTAU+RFCBAM) |
|:---|:---:|:---:|:---:|
| **MovingMNIST** (Custom Loss) | 0.7375 | **0.7114** | 0.7880 |
| **UCF101** (Custom Loss) | 0.1446 | 0.1455 | **0.1437** |
| **MovingMNIST** (MSE only) | 0.4805 | **0.4684** | 0.5092 |
| **UCF101** (MSE only) | 0.1398 | 0.1438 | **0.1389** |

**Key observations:**

- On **MovingMNIST**, model type 1 (TAU + CBAM) achieved the best performance. The combination of temporal attention with explicit spatial/channel refinement proved most effective on this simpler, synthetic dataset. The model was able to produce visually recognizable digit predictions that preserved shape and approximate motion trajectories.

- On **UCF101**, all three configurations converged to similar loss values (~0.14) after only 8 epochs of training. Model type 2 (receptive field variants) held a slight edge. However, with so few epochs on such a complex real-world dataset, the models did not develop sufficient representational capacity to faithfully reconstruct RGB video frames. The predictions on UCF101 remained blurry and lacked the fine-grained detail present in real-world videos — the task proved too complex for the limited training regime that Colab's constraints allowed.

- The **receptive field convolutions** (model type 2) came at a substantial computational cost (nearly double the per-epoch time of the baseline TAU model on UCF101) without a clear performance advantage given the constrained training budget.

### Cross-Dataset Generalization

We also evaluated generalization by training on UCF101 (3-channel RGB) and testing on MovingMNIST (1-channel grayscale), replicating the single grayscale channel to 3 channels and applying the model autoregressively in three cascaded passes:

| Model | Cross-Dataset Loss (Custom) | Cross-Dataset Loss (MSE) |
|:---:|:---:|:---:|
| 0/0 | 1.1145 | 0.9955 |
| 1/0 | 1.0828 | 0.9655 |
| 2/2 | **1.0691** | **0.9528** |

Model type 2 with receptive field attention achieved the best cross-dataset generalization, suggesting that the spatially-varying attention mechanism captures more transferable features.

---

## Key Takeaways

1. **TAU + CBAM** is the most effective combination for structured, synthetic data (MovingMNIST), where explicit spatial and channel attention complement temporal modeling.
2. **Receptive Field Attention** shows the strongest results on real-world data and cross-dataset transfer, but its computational cost is high and its advantage diminishes when training is heavily constrained.
3. **Real-world video prediction** (UCF101) remains extremely challenging: the models did not achieve sufficient quality to produce sharp, realistic frame reconstructions. More training time, larger model capacity, or complementary strategies (e.g., pretext tasks, adversarial losses) would be necessary.
4. The **custom KL-divergence loss** over temporal frame differences provides a useful learning signal for motion dynamics, though its impact is most visible on the simpler dataset.

---

## Future Work

- **Extended training** on real-world datasets with dedicated GPU infrastructure
- **Optimized receptive field convolutions** to reduce computational overhead
- **Lightweight model variants** for practical deployment
- **Integration of pretext tasks** (optical flow, depth estimation) to provide additional supervision

---

## References

1. Tan, C., Gao, Z., Li, S., Li, S. Z. (2022). *Temporal Attention Unit: Towards Efficient Spatiotemporal Predictive Learning*. [arXiv:2206.12126](https://arxiv.org/abs/2206.12126)
2. Woo, S., Park, J., Lee, J.-Y., Kweon, I. S. (2018). *CBAM: Convolutional Block Attention Module*. [arXiv:1807.06521](https://arxiv.org/abs/1807.06521v2)
3. Zhang, Y., et al. (2023). *Receptive Field Attention for Spatiotemporal Predictive Learning*. [arXiv:2304.03198](https://arxiv.org/abs/2304.03198)
