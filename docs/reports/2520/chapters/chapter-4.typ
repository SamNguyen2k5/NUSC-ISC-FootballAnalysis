= Related Works

_*U-Net as a heatmap generator*, Ronnerberger et al, 2015._ @ronneberger_u-net_2015

U-Net is a convolutional-style neural network, with a series of down-convolutional layers followed by up-convolutional layers, and 'copy-and-crop' layers that propagate pre-down-sampled features to the up-sampling layers. The network architecture is known to be robust with datasets of a limited size, with many application in segmentation or heatmap generation. In ISC-1, we used U-Net to segmentate field lines in broadcast images of football matches with decent results, and we also explored using U-Net to detect keypoints (intersection of field lines), however with poorer results. ISC-2 will mainly focus on improving U-Net's capability to detect keypoints using heatmap generation.

_*Numerical Coordinate Regression with Convolutional Neural Networks*, Nibali et al, 2018_ @nibali_numerical_2018.

Nibali et al. proposes a non-parametric differentiable spatial to numerical transform (DSNT) layer to regress a 2D heatmap into two coordinates $(x^*,y^*)$ that resembles the peak point of the heatmap by applying element-wise matrix multiplication of the predetermined kernels $bold(X),bold(Y)$ to the heatmap $bold(Z)$. Specifically, with kernels $bold(X), bold(Y)$ predetermined,

$
  x^* = ||bold(Z) dot.circle bold(X)||_F, quad
  y^* = ||bold(Z) dot.circle bold(Y)||_F
$

where $dot.circle$ is the element-wise matrix multiplication operator, and $||.||_F$ is the matrix Frobenius norm. As the layer consists of differentiable matrix operations, DSNT effectively changes the problem from heatmap regression to coordinate regression, which we can utilise simple loss functions such as L1-Loss, L2-Loss, etc., though a secondary loss function on the heatmap is still needed.

_*Focal Loss for Dense Object Detection*, Lin et al, 2018_ @lin_focal_2018

When regressing heatmaps, we notice that there is a class imbalance, with a bias towards black pixels. Typical loss functions such as $L_2$ or BinaryCrossEntropy penalise wrongly predicted white pixels as much as wrongly predicted black pixels, therefore will just incentivise the model to produce a blank heatmap of all black pixels. Lin et al's @lin_focal_2018 $"FocalLoss"$ aims to mitigate this class imbalance issue. Recall that the $"BinaryCrossEntropy"$ loss is defined as

$
  "BCE" = -sum_(t in T) y_t log hat(y)_t
$

$"FocalLoss"$ introduces the term $(1-hat(y)_t)^gamma$ to decrease the weightage of the loss term at class $t$ when the predicted probability $hat(y)_t$ gets close to $1$.
$
  "FL" = -sum_(t in T) y_t dot alpha_t (1-hat(y)_t)^gamma log hat(y)_t
$

_Heatmap Distribution Matching for Human Pose Estimation, Liu et al, 2018_ @qu_heatmap_nodate.

Liu et al proposes improvements on heatmap regression methods using the framework of _heatmap distribution matching_ by setting a loss function on the predicted heatmap and the actual Gaussian-smoothed heatmap. Pixel-wise aggregate losses such as the Mean Squared Error (MSE) do not always guide the heatmap towards localising the correct position, but can instead generate heatmaps that mislead the chosen position after $"argmax"$. The paper uses the 'Earth Mover's Distance' to compare the predicted and the annotated heatmap in the distribution sense, analogously to other distribution distances such as the Kullback-Leibler distance. Although we will not study this paper in detail, this gives us decent justifications on _heatmap matching_ approaches.
