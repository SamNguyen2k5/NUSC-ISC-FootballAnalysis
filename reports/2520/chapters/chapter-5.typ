= Methodology

#figure(
  caption: [Full pipeline of the `LocationDetector` module. The red parts indicate trainable components, whilst other components are non-trainable and are only avaiable upon evaluation.],
)[
  #image("../method/method.svg")
]

We define the `PlayerDetector` model as a composition of submodules with two main stages. The first stage is to generate a Gaussian heatmap beased on the location of the players. The model for this first stage is subject to a heatmap matching task and is available for training. The second stage is to convert the Gaussian heatmap into a collection of keypoints, which we will utilise the Bayesian Gaussian Mixture Model (BGMM) algorithm to compute the components' centre. This module is non-trainable.

== Training procedure

For the heatmap generation model, we first use the `masking_unet` module to segment the inner section of the field from the outer sections (such as the audience's area, the sky, etc.). To train `masking_unet` we utilise the project during ISC-1, where the calibration dataset @noauthor_soccernet_2025 is augmented so that the area inside the field lines are captured. The images are annotated such that a point in the image coordinates system $(x,y)$ is deemed to lie inside the playing field area if its world coordniates equivalent lies in the playing area, which is bounded by a rectangular area. The conversion can be done in batches, i.e.

$
  mat(x^("world")_1, x^("world")_2, dots, x^("world")_n; y^("world")_1, y^("world")_2, dots.c.h, y^("world")_n; lambda_1, lambda_2, dots.c.h, lambda_n)
  =
  bold(H)^(-1) dot mat(x_1, x_2, dots, x_n; y_1, y_2, dots.c.h, y_n; 1, 1, dots.c.h, 1)
$

where the construction of the homographic transformation $bold(H)$ is described in details in ISC-1. A point $(x,y)$ is in the field iff. $|x^("world")| <= W"/"2$ and $|y^("world")| <= H"/"2$, where $W times H$ are the predetermined dimensions of the field. For this project, we choose $W=110$ and $H=55$.

`heatmap_unet` is a generic U-Net that accepts a multi-channel image and produces a single-channel heatmap. To ensure that the heatmap is incentivised to predict actual players' locations, we also introduce another `gaussiniser` layer. The key observation here is that the heatmap does not represent a true Gaussian heatmap; the `gaussianiser` is there to 'convert' the heatmap into a more Gaussian-like heatmap. It ranks the pixels by intensity and takes the top-$k$ pixels, each becomes the centre of a Gaussian blob of size $sigma$ on the new heatmap. Some additional considerations are made for more robust training:

- *Sampling top-$k$ keypoints*: During experiments we choose $k=1000$ to capture as many keypoints in different areas as possible. As $k$ gets large, the training speed might be slowed down. However a small value $k$ tends to pick up neighbouring or nearby pixels rather than keypoints at different areas. Therefore we adapt a sampling approach: choose a random subset of $k dot p$ keypoints in the top-$k$ pixels, where $p = 0.1$.


- *Gaussianised heatmap overlay*: We also adapt the Gaussianiser model as an overlay on the original heatmap, with a coefficient $lambda$. It is also noticed that training with a higher $lambda$ incentives the model towards concentrating the heatmap around a few keypoints. Therefore we only use a small $lambda$ for the `gaussiniser` during training to allow the model to learn from mistakes in other non-top$k$ pixels in the original heatmap, and during inference we can set $lambda = 1$.

#figure(
  caption: [Original heatmap generated from `heatmap_unet`; Gaussianised heatmap with $lambda=0.1$; Gaussianised heatmap with top-$k$ sampling and $lambda = 1$ _(from left to right)_],
  placement: bottom,
)[
  #grid(
    columns: 3,
    image("../method/gaussianiser_pre.png"),
    image("../method/gaussianiser_soft.png"),
    image("../method/gaussianiser_hard.png"),
  )
]

#figure(
  caption: [Gaussianised heatmap with $lambda=0.5$ during training. The top-$k$ pixels are concentrated around a single keypoint.],
  placement: top,
)[
  #image("../method/gaussianiser_too_hard.png")
]

The heatmap matching task is trained on $"FocalLoss"$ and our self-defined loss function called the $"IntensityLoss"$. The $"IntensityLoss"$ is defined as the $"F1Loss"$ of the mean pixel intensity of the predicted heatmap and the ground-truth heatmap, which forces the model to concentrate the bright pixels around certain points rather than spreading the intensity throughout the whole heatmap. The $"FocalLoss"$ follows Lin et al's @lin_focal_2018 definition, with our chosen parameters as $alpha_1=0.99$ is the coefficient for the white-pixel class. and $gamma=2$ as the power parameter for the $(1-hat(y)_1)^gamma$ term.

== Inference procedure

The Gaussianised heatmap is then fed into a Bayesian Gaussian Mixture Model (BGMM) to extract the centroids which then become our keypoints. From the heatmap, we resample the pixels to collect a list of pixels $(x,y)$ with a probability proportionate the to the pixel's intensity, which is then fed into the BGMM model. We set the `max_components=30`, then remove all predicted components whose `weights` are less than a predetermined `tolerance=0.01`. This ensures that components with very low weightage are removed from the final output, these components are usually covered by another component of a higher weightage. The keypoints are then transformed into world coordinates using the reverse process of the undistort-and-inverse-homography transformation described in @section:task.

A full inference pipeline is available in the inference test Jupyter Notebok at `experiments/ex06_player_heatmap_after_masking/spiideo-inference-test.ipynb`.
