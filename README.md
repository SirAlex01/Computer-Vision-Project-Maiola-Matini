*ABSTRACT*

Video frames prediction is a complex task, that only during the last years has seen some solutions and architectures for it. In this project, we researched a solution to the problem that combines 
many of the stepping stones that were placed in the last years. The solution was born out of much trial and error, where many other components and subarchitectures were tried and discarded, before 
finally arriving at this solution. In particular, architectures from three papers were combined to form our solution: the TemporalAttentionUnit modules (TAU) from the paper at https://arxiv.org/abs/2306.11249,
the Convolutional Block Attention Module (CBAM) technique to enhance the spatial awareness of the network from the paper at https://arxiv.org/abs/1807.06521v2, and finally the receptive fields attention idea,
which helped with performances in both the spatial and temporal prediction core areas, deriving from the paper at https://arxiv.org/abs/2304.03198. We present in particular many ways of mixing and putting 
together these blocks to create different architectures, that vary in where and if our techniques are used and the attention modules' positioning with respect to each other.
In any case, such attention layer is always preceded by an Encoder and succeeded by a Decoder that act in a U-Net fashion.
