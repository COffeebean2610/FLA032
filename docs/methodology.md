# Research Methodology & Technical Formulation

## 1. Mathematical Formulation of the Classification Problem
Let the input space be defined by the set of all digital foliage photographs $\mathcal{X} \subset \mathbb{R}^{H \times W \times 3}$, where $H = W = 224$ denotes the standardized spatial height and width, and 3 denotes the RGB color channels.
Let the target label space be $\mathcal{Y} = \{1, 2, \dots, C\}$ where $C = 44$ represents the set of mutually exclusive crop-pathology categories.
The objective is to learn a parameterized mapping function $f_\theta: \mathcal{X} \to \Delta^{C-1}$ where $\Delta^{C-1}$ represents the probability simplex:
$$\hat{y} = \arg\max_{c \in \{1, \dots, C\}} P(Y = c \mid X; \theta) = \arg\max_{c \in \{1, \dots, C\}} \frac{\exp(z_c)}{\sum_{j=1}^C \exp(z_j)}$$
where $z_c$ denotes the logit output corresponding to class $c$.

The objective function optimized during training is the multi-class categorical cross-entropy loss:
$$\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log \hat{y}_{i,c}$$
where $y_{i,c} \in \{0, 1\}$ is the one-hot ground-truth indicator for sample $i$ and class $c$.

---

## 2. Deep Learning Architectures

### Architecture 1: Custom CNN Baseline
A 4-stage convolutional neural network engineered from first principles:
- **Feature Extraction Stages:**
  - Stage 1: Conv2D(32 filters, 3×3, stride 1) $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool(2×2)
  - Stage 2: Conv2D(64 filters, 3×3, stride 1) $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool(2×2)
  - Stage 3: Conv2D(128 filters, 3×3, stride 1) $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool(2×2)
  - Stage 4: Conv2D(256 filters, 3×3, stride 1) $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool(2×2)
- **Classification Head:**
  - GlobalAveragePooling2D $\to$ Dense(256, ReLU) $\to$ BatchNorm $\to$ Dropout($p = 0.40$) $\to$ Dense(44, Softmax).

### Architecture 2: VGG16 Transfer Learning
- Utilizes the 16-layer architecture introduced by Simonyan & Zisserman (2014) pre-trained on ImageNet.
- Characterized by homogeneous 3×3 convolutions organized into 5 convolutional blocks.
- Backbone weights frozen during initial stage; top feature maps aggregated via Global Average Pooling into a 256-dimensional dense embedding before classification.

### Architecture 3: ResNet50 Residual Learning
- Utilizes the 50-layer deep residual network by He et al. (2016).
- Addresses vanishing gradients via identity shortcut connections:
  $$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$
- Enables deep hierarchical feature representations through bottleneck residual building blocks.

### Architecture 4: MobileNetV2 (Edge-Optimized Architecture)
- Utilizes the inverted residual and linear bottleneck architecture developed by Sandler et al. (2018).
- Decomposes standard convolution into Depthwise Separable Convolutions:
  $$\text{Depthwise}(W_d, X) + \text{Pointwise}(W_p, X)$$
  reducing computational cost by approximately a factor of $8$ to $9$ relative to standard 3×3 convolutions:
  $$\frac{D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F + M \cdot N \cdot D_F \cdot D_F}{D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F} = \frac{1}{N} + \frac{1}{D_K^2}$$
- Highly optimized for mobile and micro-edge inference.

---

## 3. Explainable AI: Grad-CAM Formulation
Grad-CAM (Selvaraju et al., 2017) provides visual explanations by calculating the gradient of the predicted class score $y^c$ with respect to feature activation map $A^k$ of the final convolutional layer:
$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial y^c}{\partial A_{i,j}^k}$$
where $Z = u \times v$ represents the spatial dimensions of the feature map.
The class-discriminative saliency heatmap is formed as the rectified linear combination of feature maps:
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
The $\text{ReLU}$ ensures that only features having a positive correlation with the target class are highlighted.

---

## 4. Edge Deployment and Quantization
Post-training quantization reduces storage footprint and memory bandwidth requirements:
1. **Float32 Baseline:** Unquantized IEEE 754 32-bit floating point weights and activations.
2. **Float16 Quantization:** Constants/weights converted to 16-bit half-precision floating point numbers.
3. **Integer (INT8) Quantization:**
   Maps continuous real-valued weights $r \in [\alpha, \beta]$ into discrete 8-bit signed integers $q \in [-128, 127]$:
   $$r = S \cdot (q - Z)$$
   where scale factor $S = \frac{\beta - \alpha}{255}$ and zero-point $Z = \text{round}\left(-\frac{\alpha}{S}\right) - 128$.
