*ABSTRACT*

Video frame prediction is a very challenging task that has seen significant advancements in recent years. Many solutions utilize pretext tasks, such as Depth Estimation or Optical Flow, to improve performance. However, some approaches focus solely on using raw RGB or grayscale video frames, avoiding additional data preprocessing. Our work is centered on this latter approach, building on recent advancements in this area. We use the architecture proposed in the paper https://arxiv.org/abs/2306.11249, which introduced an Attention Block for capturing temporal dependencies between frames.

Our project integrates several recent advancements by combining various architectures. After extensive experimentation and iteration, we settled on a final design that merges elements from three key papers: the Temporal Attention Unit (TAU) from the aforementioned paper, the Convolutional Block Attention Module (CBAM) for enhancing spatial awareness https://arxiv.org/abs/1807.06521v2, and the receptive fields attention concept for improving both spatial and temporal prediction https://arxiv.org/abs/2304.03198. We experimented with different configurations of these components, varying their integration and positioning within our model. Each attention layer is preceded by an Encoder and followed by a Decoder in a U-Net style architecture.

To ensure the reproducibility of our results, we implemented an experimental setup that supports continuous training and checkpoint collection of our models, optimizing for validation loss. We evaluated our approach on two distinct datasets:

UCF101: A dataset designed for action recognition with real-life RGB videos featuring 101 different human actions.
MovingMNIST: A synthetic dataset consisting of grayscale videos.
Additionally, we assessed the generalization capability of our network by training on the UCF101 dataset and testing it with MovingMNIST videos, despite challenges due to the differing input channels. We present and discuss the results of these experiments.
