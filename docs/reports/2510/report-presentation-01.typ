#import "@preview/slydst:0.1.4": *
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node

#set text(size: 10pt)
#set par(leading: 0.8em, justify: true)

#set math.mat(delim: "[")
#let vec(v) = { $ harpoon(bold(#v)) $ }
#let hvec(v) = { $ tilde(bold(#v)) $ }
#let unit(v) = { $ hat(bold(#v)) $ }

#show: slides.with(
  title: [Theoretical Foundations for 3D Reconstruction of Sports Games with Limited Visual Data Settings],
  subtitle: [Independent Study Course (NST3901)],
  date: none,
  authors: ("Nguyen Hoang The Kiet",),
  layout: "medium",
  ratio: 16 / 10,
  title-color: none,
)

#let colored-quote(
  fill: luma(230),
  body,
) = block(
  width: 100%,
  fill: fill,
  inset: 16pt,
  radius: 4pt,
  spacing: 10pt,
)[
  #body
]

== Table of Contents
#outline()

= Motivation

== Semi-automated offside technology
#figure()[
  #grid(
    columns: (50%, 50%),
    column-gutter: 3pt,
    image("motivation/serie-a.png", height: 100%, fit: "contain"),
    image("motivation/serie-a-offside.png", height: 100%, fit: "contain"),
  )
]

== VAR controversies in the 2024 ASEAN Cup @asean_united_fc__2024 @asean_united_fc__2024-1
#figure()[
  #grid(
    columns: (50%, 50%),
    column-gutter: 3pt,
    image("motivation/asean-cup-phi-tha.png", height: 100%, fit: "contain"),
    image("motivation/asean-cup-tha-phi.png", height: 100%, fit: "contain"),
  )
]


= A suggested pipeline

== 3-stage pipeline

#figure(
  caption: [Summary of the suggested automated pipeline for field registration],
)[
  #set text(9pt)
  #set par(leading: 0.6em)
  #diagram(
    spacing: (18mm, 10mm),
    node-stroke: luma(80%),

    node(
      (0, -0.75),
      [_SoccerNet Camera \  Calibration dataset_ @noauthor_soccernet_2025],
      name: <dataset>,
      fill: white,
    ),

    node((0, 0), [*Raw monochrome image* \ $H times W$], name: <raw_image>, fill: white),
    node((0, 1), [*Segmented image mask* \ $H times W$], name: <seg_mask>, fill: white),
    node((0, 2), [*Keypoint concentration mask* \ $H times W$], name: <keypoint_mask>, fill: white),

    node((1.25, 2), [*Graph nodes* \ $V times 3$], name: <nodes>, fill: white),
    node((1.25, 1), [*Graph edges* \ $E times 3$], name: <edges>, fill: white),
    node((1.25, -0.5), [*Groups of \ orthogonal lines*], name: <lines>, fill: white),

    node((2.75, -0.5), [*Vanishing points* \ $2 times 3$], name: <vanishing_pt>, fill: white),
    node((2.75, 1), [*Vanishing line* \ $1 times 3$], name: <vanishing_line>, fill: white),
    node((2.75, 2), [*Homography \ transformation* \ $3 times 3$], name: <hom>, fill: white),

    edge(<dataset>, <raw_image>, "->", []),
    edge(<raw_image>, <seg_mask>, "->", [_UNet_]),
    edge(<seg_mask>, <keypoint_mask>, "->", [_UNet_]),
    edge(<seg_mask>, <edges>, "->", [_Sampling_]),
    edge(<keypoint_mask>, <nodes>, "->", [_Clustering_]),
    edge(<nodes>, <edges>, "->", [_Sampling_]),
    edge(<edges>, <lines>, "->", [_Clustering \ of gradients_], label-side: left),
    edge(<lines>, <vanishing_pt>, "->", [_SVD_]),
    edge(<vanishing_pt>, <vanishing_line>, "->", [_Cross product_]),
    edge(<lines>, <hom>, "->", [_Rectification_]),
    edge(<vanishing_pt>, <hom>, "->", [], label-side: left),
    edge(<edges>, <hom>, "->", []),
    edge(<nodes>, <hom>, "->", []),

    node(
      // text(orange)[Node group],
      fill: orange.transparentize(90%),
      stroke: orange.transparentize(60%),
      corner-radius: 12pt,
      enclose: (<raw_image>, <seg_mask>, <keypoint_mask>),
      name: <group_1>,
      layer: -100,
    ),

    node(
      // text(orange)[Node group],
      fill: yellow.transparentize(90%),
      stroke: yellow.transparentize(60%),
      corner-radius: 12pt,
      enclose: (<nodes>, <edges>, <lines>),
      name: <group_1>,
      layer: -100,
    ),

    node(
      // text(orange)[Node group],
      fill: red.transparentize(90%),
      stroke: red.transparentize(60%),
      corner-radius: 12pt,
      enclose: (<vanishing_line>, <vanishing_pt>, <hom>),
      name: <group_1>,
      layer: -100,
    ),
  )
]

== 3-stage pipeline
- *Feature masks generation*.
  - Generative image models (U-Net @ronneberger_u-net_2015)
  - Gray-scale image masks

- *Graph creation*.
  - Keypoints, field lines $==>$ Graph
  - *Orthogonal* field lines
  - Robustness (gamma correction, sampling)

- *Homography reconstruction*.
  - Projective / Multiview geometry @hartley_multiple_2003
  - Linear algebra

== Dataset
#figure()[
  #image("prep/annotations.png", height: 45%) \
  #image("rectification/rectification-known.png", height: 45%),
]

#table(
  columns: (42%, 21%, 40%),
  align: center,
  table.header([*Task / `Dataset` name*], [*Input*], [*Output / Labels*]),
  [_General_ \ `CalibrationDataset`], [Raw image \ $(3 times H times W)$], [Raw `JSON` dictionary],

  [_Edge segmentation_ \ `MaskingCalibrationDataset`],
  [Grayscaled image \ $(H times W)$],
  [Image mask, highlighted field lines \ $(H times W)$],

  [_Keypoint concentration_ \ `MaskingCalibrationDataset` \ `(keypoint_only=True)`],
  [Grayscaled image \ $(H times W)$],
  [Image mask, highlighted keypoints \ $(H times W)$],

  [_Keypoints_ \ `PitchKeypointCalibrationDataset`],
  [Grayscaled image \ $(H times W)$],
  [List of all keypoints, reconstructed via homography \ $(27 times 3)$],

  [_Homography reconstruction_ \ `PitchHomographyCalibrationDataset`],
  [Grayscaled image \ $(H times W)$],
  [The homography matrix \ $(3 times 3)$],
)

= Stage 1. Image segmentation
== Segmentation of field lines and keypoints
#figure()[
  #image("unet/unet-seg-target.png")
]

== UNet @ronneberger_u-net_2015: A generative model for feature extraction
#grid(
  columns: (50%, 50%),
  [
    - Strong in *localisation* tasks
      - Convolutional layers
      - Detecting patterns based on the relationship between neighbouring pixels

    - Skipping neurons
      - Perserves global properties

    - Won ISBI 2015 EM segmentation challenge in low-data conditions
      - Ability to learn even with the lack of data
  ],
  figure(caption: [UNet's architecture @ronneberger_u-net_2015])[
    #image("unet/unet.png")
  ],
)

== Choosing a loss function for segmentation tasks @noauthor_loss_nodate

- Cross Binary Entropy Loss
$
  cal(L)_"BCE" (vec(x), vec(y)) = - sum_(i) log(x_i y_i) = - log(vec(x)) dot log(vec(y))
$

- Intersection-over-Union (IOU) Loss
$
  cal(L)_"IOU" (vec(x), vec(y)) = (|X inter Y|)/(|X union Y|) = (vec(x) dot vec(y))/(1 - (vec(bb(1)) - vec(x))(vec(bb(1)) - vec(y)))
$

#colored-quote(fill: orange.transparentize(75%))[
  - Dice Loss (for binary classes)
  $
    cal(L)_"Dice" (vec(x), vec(y)) = (2|X inter Y|)/(|X| + |Y|) = 2 dot (vec(x) dot vec(y))/(vec(x) dot vec(bb(1)) + vec(y) dot vec(bb(1)))
  $
]

== Training metrics (field lines, `n_epochs = 64`)
#figure()[
  #image("unet/unet-seg-metric.png")
]

== Qualitative results (field lines, `epoch = 05`)
#figure()[
  #image("unet/unet-seg-training-e05.png")
]

== Qualitative results (field lines, `epoch = 13`)
#figure()[
  #image("unet/unet-seg-training-e13.png")
]

== Qualitative results (field lines, `epoch = 28`)
#figure()[
  #image("unet/unet-seg-training-e28.png")
]

== Qualitative results (field lines, `epoch = 46`)
#figure()[
  #image("unet/unet-seg-training-e46.png")
]

== Qualitative results (field lines, `epoch = 64`)
#figure()[
  #image("unet/unet-seg-training.png", width: 120%)
]

== Training metrics (keypoints, `n_epochs = 48`)
#figure()[
  #image("unet/unet-kp-metric.png")
]

== Qualitative results (field lines)

= Stage 2. Graph creation

== Keypoint detection via pixel clustering

#figure(
  // caption: [],
)[
  #grid(
    columns: (50%, 50%),
    image("masks/image-segmentation-mask-plain.png"), image("masks/image-keypoint-mask-plain.png"),
  )
]

== Keypoint detection via pixel clustering

#figure(
  // caption: [],
)[
  #grid(
    columns: (50%, 50%),
    image("masks/image-segmentation-mask.png"), image("masks/image-keypoint-mask.png"),
  )
]


== Keypoint detection via pixel clustering

#figure()[
  #image("clustering/keypoint-clustering.svg", height: 100%)
]

== Edge sampling

#figure()[
  #grid(
    columns: (63%, 36%),
    image("clustering/edge-sampling.svg", width: 75%),
    [
      #image("clustering/gamma-correction.png") \
      Gamma correction plot, $gamma = 0.05$
    ],
  )
]

- Edge mask $I_E$ of size $H times W$;
- Two keypoints $vec(p)_1$ and $vec(p)_2$;
- $gamma approx 0.01$
#colored-quote(fill: yellow.transparentize(66%))[
  Step 1. Let $T := ceil(||vec(p)_1 - vec(p)_2||)$.

  Step 2. Generate $T$ random values of $t^((i)) in [0, 1]$.

  Step 3. For each $t^((i))$, the sampled point is $vec(q)^((i)) := (1 - t^((i))) vec(p)_1 + t^((i)) vec(p)_2 + vec(bold(epsilon))$, where $vec(bold(epsilon))$ is a random noise vector, taking range $[-2, 2]$ for each dimension.

  Step 4. Return confidence as the sum of the points' intensities raised to the power of $gamma$:
  $ "Confidence" := 1/T sum_i I_E (q^((i))_x, q^((i))_y)^gamma $
]

== Clustering of gradients

#figure()[
  #image("clustering/edge-detected.png", width: 75%) \
]

#figure()[
  #image("clustering/gradients-clustering.svg") \
]

#colored-quote(fill: yellow.transparentize(66%))[
  *Grouping lines by the gradient vector*.

  _Input_.
  - Set of lines $L = {hvec(l)^((i))}$

  _Output_. Two sets of parallel lines $L_1$ and $L_2$, where each line in $L_1$ is orthogonal with a line in $L_2$ in the original space.

  Step 1. For each line $hvec(l)^((i))$, compute the normalised gradient vector $display(unit(v)^((i)) = 1/(sqrt(l_1^(i)^2 + l_2^(i)^2)) mat(l_1^((i)); l_2^((i))))$

  Step 2. Use a clustering algorithm $cal(C)$ to group the lines into two groups.
]

#figure()[
  #image("clustering/gradients-clustering-on-circle.png") \
]

= Stage 3. Homography reconstruction

== SoccerNet camera model @noauthor_soccernet_2025

#figure()[
  #image("prep/field.png")
]

== Projective geometry

#figure()[
  #image("transforms-2d/projective.svg", height: 100%)
]

- Homogenous representation of a point and a line:
$ hvec(p) = mat(x; y; z); hvec(l) = mat(a; b; c) $

- Incidence:
$ (p) in (ell) <=> hvec(l)^T hvec(x) = 0 $

- Line passing two points:
$ hvec(p) = hvec(l)_1 times hvec(l)_2 $

- Intersection of two lines:
$ hvec(l) = hvec(x)_1 times hvec(x)_2 $

== Transformations in 2D

#grid(
  columns: (33%, 33%, 33%),
  column-gutter: 10pt,
  align: center,
  image("transforms-2d/similarity.svg"), image("transforms-2d/affinity.svg"), image("transforms-2d/homography.svg"),
  [
    $ hvec(p)' = mat(k cos theta, -k sin theta, v_x; k sin theta, k cos theta, v_y; 0, 0, 1) hvec(p) $ \ *Similarity*
  ],
  [
    $ hvec(p)' = mat(a_11, a_12, v_x; a_21, a_22, v_y; 0, 0, 1) hvec(p) $ \ *Affinity*
  ],
  [
    $ hvec(p)' = mat(h_11, h_12, h_13; h_21, h_22, h_23; h_31, h_32, h_33) hvec(p) $ \ *Homography*
  ],
)

== Homography reconstruction: Known correspondences

#image("rectification/rectification-known.png"),

- Recover as many unseen keypoints as possible
- Direct Linear Transform (DLT) algorithm (_The 4-point algorithm_)

== Homography reconstruction: Unknown correspondences

#grid(
  columns: (60%, 40%),
  column-gutter: 10pt,
  image("rectification/rectification.png"),
  [
    \
    - Homography rectification
    - Affine rectification
    - Similarity rectification
    - RANSAC
  ],
)

== Homography reconstruction: Known correspondences

- Find a homography $H$ given a set of points $P = {hvec(p)^((i))}$ and $Q = {hvec(q)^((i))}$, and that for all $i$:
$
  H: hvec(p)^((i)) mapsto hvec(q)^((i))
$

- Need at least four point to determine such homography
- If more than four points $-->$ least squares approach $-->$ SVD

$
      & H hvec(p)^((i)) times hvec(q)^((i)) = hvec(0) \
  // <=>
  // & mat(delim: "|",
  //   hvec(i), hvec(j), hvec(k);
  //   h_11 p^((i))_1 + h_12 p^((i))_2 + h_13 p^((i))_3,
  //   h_21 p^((i))_1 + h_22 p^((i))_2 + h_23 p^((i))_3,
  //   h_31 p^((i))_1 + h_32 p^((i))_2 + h_33 p^((i))_3;
  //   q^((i))_1, q^((i))_2, q^((i))_3;
  // ) = 0 \
  <=> & cases(
          (h_11 p^((i))_1 + h_12 p^((i))_2 + h_13 p^((i))_3) q^((i))_2
          - (h_21 p^((i))_1 + h_22 p^((i))_2 + h_23 p^((i))_3) q^((i))_1 = 0,
          (h_21 p^((i))_1 + h_22 p^((i))_2 + h_23 p^((i))_3) q^((i))_3
          - (h_31 p^((i))_1 + h_32 p^((i))_2 + h_33 p^((i))_3) q^((i))_2 = 0,
          (h_31 p^((i))_1 + h_32 p^((i))_2 + h_33 p^((i))_3) q^((i))_1
          - (h_11 p^((i))_1 + h_12 p^((i))_2 + h_13 p^((i))_3) q^((i))_3 = 0
        ) \
  <=> & mat(
          p^((i))_1 q^((i))_2, p^((i))_2 q^((i))_2, p^((i))_3 q^((i))_2, - p^((i))_1 q^((i))_1, - p^((i))_2 q^((i))_1, - p^((i))_3 q^((i))_1;
          , , , p^((i))_1 q^((i))_3, p^((i))_2 q^((i))_3, p^((i))_3 q^((i))_3, - p^((i))_1 q^((i))_2, - p^((i))_2 q^((i))_2, - p^((i))_3 q^((i))_2;
          - p^((i))_1 q^((i))_3, - p^((i))_2 q^((i))_3, - p^((i))_3 q^((i))_3, , , , p^((i))_1 q^((i))_1, p^((i))_2 q^((i))_1, p^((i))_3 q^((i))_1;
        )
        mat(h_11; h_12; h_13; dots.v; h_31; h_32; h_33)
        = mat(0; 0; 0) \
  <=> & A^((i)) hvec(h) = hvec(0)
$

== Homography reconstruction: Unknown correspondences
#image("rectification/similarity-rectification.png", width: 100%),
#grid(
  columns: (50%, 50%),
  align: center,
  [Affine and Metric rectification], [Similarity rectification],
)

== Homography reconstruction: Affine rectification

== Homography reconstruction: Metric rectification

== Homography reconstruction: Similarity rectification
- Find a similarity $H_S$ that maps a line segment in the source $P$ to a line segment in the target $Q$.
- The mapped line segment in $P$ might be the whole segment or a _subsegment_ in $Q$.

- Only need *one* correspondence of segments from $P$ to $Q$ $=>$ exhaustive search.
- Which correspondence is the best performing one? How should a correspondence be evaluated?

== Homography reconstruction: Similarity rectification
#colored-quote(fill: red.transparentize(66%))[
  *Similarity rectification from a correspondence of two segments*.

  - _Input_: Four points $vec(p)^((1)), vec(p)^((2)), vec(q)^((1)), vec(q)^((2))$ indicating two segments $vec(p)^((1)), vec(p)^((2))$ and $vec(q)^((1)), vec(q)^((2))$.
  - _Output_: The similarity $H_S$.

  _Step 1_. Let $(p_1^((i)), p_2^((i)), 1) := hvec(p)^((i))$, $(q_1^((i)), q_2^((i)), 1) := hvec(q)^((i))$ for $i=1,2$.

  _Step 2_. Solve the linear system $H_S hvec(p)^((i)) = hvec(q)^((i))$.
  $
    cases(
      a p_1^((1)) - b p_2^((1)) + c = q_1^((1)),
      a p_2^((1)) + b p_1^((1)) + d = q_2^((1)),
      a p_1^((2)) - b p_2^((2)) + c = q_1^((2)),
      a p_2^((2)) + b p_1^((2)) + d = q_2^((2))
    )
    quad therefore quad
    mat(
      p_1^((1)), - p_2^((1)), 1, 0;
      p_2^((1)), p_1^((1)), 0, 1;
      p_1^((2)), - p_2^((2)), 1, 0;
      p_2^((2)), p_1^((2)), 0, 1
    )
    mat(a; b; c; d) = mat(
      q_1^((1)); q_2^((1));
      q_1^((2)); q_2^((2))
    )
  $

  The similarity can thus be solved using any linear solver such as the SVD.
]

== Homography reconstruction: Similarity rectification
#figure()[
  #image("rectification/inlier.svg", width: 60%)
]


== Homography reconstruction: Similarity rectification
#colored-quote(fill: red.transparentize(66%))[
  *Modified RANSAC to find segment correspondence*.

  // - _Input_:
  //   - All segments
  //   - A threshold $T$
  // - _Output_:

  _Step 1_. Let $S_P := emptyset$ and $S_Q := emptyset$

  _Step 2_. Select a random segment $bold(p)^((alpha beta)) = (vec(p)^((alpha)), vec(p)^((beta)))$ in $bold(P)$ and a random segment $bold(q)^((gamma delta)) = (vec(q)^((gamma)), vec(q)^((delta)))$.

  _Step 3_. Find a similarity $H_S$ such that the segment $bold(p)^((alpha beta))$ is mapped to $bold(q)^((gamma delta))$.

  _Step 4_. For each $bold(p)^((a b))$ in $bold(P)$ and $bold(q)^((c d))$, calculate the distances between the endpoints and the other segment:
  $
    d_a := op("dist")(vec(p)^((a)), bold(q)^((c d))); quad quad
    d_b := op("dist")(vec(p)^((b)), bold(q)^((c d)))
  $

  - Let $display(d_"avg" = 1/2 (d_a + d_b))$. If $d_"avg" <= T$, $S_P := S_P union {a b}, S_Q := S_Q union {c d}$.

  _Step 5_. Let $"Score" = |S_P| + |S_Q|$. If $"Score" >= "Score"_"Accept"$, return $H_S$. Otherwise, repeat the process.
]

== Results
#image("rectification/results.png")
#image("../../exports/251108/pipeline/00126.jpg")
#image("../../exports/251108/pipeline/01210.jpg")
#image("../../exports/251108/pipeline/00132.jpg")
#image("../../exports/251108/pipeline/02094.jpg")

== Discussion
- *Feature masks generation*.
  - Loss function (Regularisation, geometric invariants, etc.)
  - Data augmentation techniques (constrast, brightness, etc.)

- *Graph creation*.
  - More robust algorithms
  - Improving speed (bottleneck at RANSAC)

- *Homography reconstruction*.
  - Numerical stability
  - Different representation of the homography transform (4-point, camera displacement, etc.)

== References
#bibliography("refs.bib")
