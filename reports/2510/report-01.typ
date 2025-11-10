// #import "@preview/bamdone-ieeeconf:0.1.1": ieee
#import "@preview/basic-report:0.3.1": *
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node

#show: it => basic-report(
  doc-category: "Independent Study Course (NST3901)",
  doc-title: [Theoretical Foundations for 3D Reconstruction of Sports Games with Limited Visual Data Settings],
  author: "Nguyen Hoang The Kiet",
  affiliation: "NUS College, National University of Singapore",
  // logo: image("assets/aerospace-engineering.png", width: 2cm),
  // <a href="https://www.flaticon.com/free-icons/aerospace" title="aerospace icons">Aerospace icons created by gravisio - Flaticon</a>
  language: "en",
  // compact-mode: true,
  it,
)

// #show: word-count

#set page(
  paper: "a4",
  margin: 2cm,
  background: rotate(24deg, text(15pt, fill: rgb("#ff3f2145"))[
    * DRAFT * \
    * DRAFT * \
    * DRAFT *
  ]),
)
#set text(size: 12pt)
#set par(leading: 0.8em, justify: true)
#set quote(block: true)
#set math.mat(delim: "[")

#show heading: set block(
  above: 2em,
  below: 1em,
)

#show quote: set block(
  fill: luma(230),
  inset: 8pt,
  radius: 4pt,
  spacing: 10pt,
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

#set math.mat(delim: "[")
#let vec(v) = { $ harpoon(bold(#v)) $ }
#let hvec(v) = { $ tilde(bold(#v)) $ }
#let unit(v) = { $ hat(bold(#v)) $ }

#let box(content) = rect(
  stroke: 0.8pt + black,
  radius: 2pt,
  width: 2.2cm,
  height: 0.8cm,
  align(center, content),
)

// Your content goes below.

= Introduction

== Motivation #footnote[Adopted from the ISC application form]

The Video Assistant Referee (VAR) system was introduced at the 2018 FIFA World Cup in Russia, after which it became popularised in other European domestic and continental leagues. Originally a video review system for referees in case of critical fouls (e.g., offsides leading to a goal, offences in the penalty box, etc.), the technology has been equipped with major upgrades, such as the semi-automated offside technology (SAOT) used in the latest edition of the FIFA World Cup in Qatar @inside_fifa_semi-automated_2022.

#figure(
  caption: [Semi-automated offside technology used in Italy’s Serie A.],
)[

]

Lower-tier tournaments such as the 2024 ASEAN Cup (formerly known as AFF Cup) and domestic leagues in Southeast Asia are also improving transparency in refereeing decisions by implementing low-budget versions of the standard Video Assistant Referee (VAR), such as VAR Lite or centralised VAR. However, these budget systems quickly showed vulnerability in operation, especially in high-stakes situations. For instance, in the two legs of the Thailand vs. Philippines match-off in the 2nd semifinal of the ASEAN Cup:

- In the first leg @asean_united_fc__2024-1, VAR did not participate in a potential offside offence by the Filipino player Alex Munis, which then led to the Philippines’ first goal in their historic 2-1 win against Thailand. Speculations from social media then showed that VAR seemed not to have worked at all due to the latency between the field and the centralised VAR station in Singapore.

- In the second leg @asean_united_fc__2024, VAR also did not intervene when the Thai player Seksan Ratree allegedly dribbled the ball crossing the goal line, leading to Peradol’s opening goal for Thailand that would win them the leg and the semi-final. The replay was indecisive and caused many controversies afterwards.

#figure(
  caption: [Two of the controversial incidents in the Thailand v Philippines ASEAN Cup semi-final legs.],
)[]

In light of the mentioned shortcomings, this project aims to build an end-to-end solution for reconstructing game events in 3D under limited data conditions, with potential applications in refereeing, analytics and broadcasting.

== Scope of the Independent Study Course project

The first stage of the project will be a survey of foundational methods in the field registration stage. The aim is to reconstruct a model of a football field

== The suggested pipeline

#figure(
  caption: [Summary of the suggested automated pipeline for field registration],
)[
  #diagram(
    spacing: (18mm, 10mm),
    node-stroke: luma(80%),

    node((0, -1), [_SoccerNet Camera \  Calibration dataset_ @noauthor_soccernet_2025], name: <dataset>, fill: white),

    node((0, 0), [*Raw monochrome image* \ $H times W$], name: <raw_image>, fill: white),
    node((0, 1), [*Segmented image mask* \ $H times W$], name: <seg_mask>, fill: white),
    node((0, 2), [*Keypoint concentration mask* \ $H times W$], name: <keypoint_mask>, fill: white),

    node((1.25, 2), [*Graph nodes* \ $V times 3$], name: <nodes>, fill: white),
    node((1.25, 1), [*Graph edges* \ $E times 3$], name: <edges>, fill: white),
    node((1.25, -0.5), [*Groups of \ orthogonal lines*], name: <lines>, fill: white),

    node((3, -0.5), [*Vanishing points* \ $2 times 3$], name: <vanishing_pt>, fill: white),
    node((3, 1), [*Vanishing line* \ $1 times 3$], name: <vanishing_line>, fill: white),
    node((3, 2), [*Homography \ transformation* \ $3 times 3$], name: <hom>, fill: white),

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

We propose the above automated pipeline for the field registration problem, which can be divided into three smaller subtasks with its respective domain of knowledge:

- *Feature masks generation*. Using a generative image model (for instance, U-Net @ronneberger_u-net_2015), we seperate the important features from the raw image of the field, i.e. the field lines and keypoints, in the form of a gray-scale image mask.

- *Graph creation*. From the image masks, we detect the keypoints in coordinates form and construct the edges between them. Due to the potential noise of the masks, however, the challenge is to make the algorithms robust through techniques such as gamma correction and sampling. Finally, exploiting the orthogonal structure of the field lines, we classify each field line as either 'vertical' or 'horizontal.' This additional information will assist the following step.

- *Homography reconstruction*. From the constructed graph and the coordinates of each keypoint, we shall reconstruct a transformation from the world view (described by a template field) to the camera view (given by the image). Such a transformation from a template, planar field to the transformed field is called a homography, the process of recovering which is well described in textbooks such as @hartley_multiple_2003. However, as we do not have a correspondence keypoint-keypoint mapping, ...

= Feature masks generation via generative models
== Generative models as feature generators

The first step for field registration is to

Notice that the objects for detection, i.e. field lines, are not suitable for a object detection framework.

- Strengths of UNet @ronneberger_u-net_2015:
  - Localisation
  - Low data
  - Based on a DCN that won ISBI 2012
  - Best model for segmentation (won EM segmentation challenge at ISBI 2015)

// == YOLO-based methods
// - YOLO segmentation models

== ...



= Detecting keypoints and edges of the skeleton graph via clustering

== Detecting keypoints through clustering methods
#figure(
  caption: [Outline mask (left) and keypoint concentrated mask (right)],
)[
  #grid(
    columns: (50%, 50%),
    image("masks/image-segmentation-mask.png"), image("masks/image-keypoint-mask.png"),
  )
]


From the keypoint concentrated mask, we extract all coordinates $(x^((i)), y^((i)))$ whose pixel intensity level exceeds a threshold level $T$ ($T approx 0.8$). We shall thus obtain a list of "white" pixels $display(bold(P) = mat(vec(p)_1, vec(p)_2, dots.c.h, vec(p)_m))$. This list is yet the desired keypoints list, instead we observe from the mask that each keypoint is associated with a "cluster" of "white" pixels, which means that a keypoint is the average of some "white" points $vec(p)_i$ within its clutser.  A similar approach can be seen in @schlag_ancient_2017.

The above observation motivates us for a keypoint positioning algorithm via clustering methods as follows:

#colored-quote(fill: yellow.transparentize(66%))[
  *Keypoints detection via clustering of "white" pixels*
  - _Input_.
    - Keypoint concentrated mask $I_K$ of size $H times W$.
    - Threshold $T approx 0.8$
  - _Output_. List of keypoints
  $
    display(
      P = mat(
        x^((1)), x^((2)), dots.c.h, x^((n));
        y^((1)), y^((2)), dots.c.h, y^((n));
      )
    )
  $
  - _Hyperparameters_. A clustering algorithm $cal(C)$.

  _Step 1_.
  From the keypoint concentrated mask, extract all coordinates $(x^((i)), y^((i)))$ whose pixel intensity level exceeds a threshold level $T$, thus obtaining a list of "white" pixels $display(bold(Q) = mat(vec(q)^((1)), vec(q)^((2)), dots.c.h, vec(q)^((m))))$.

  _Step 2_. Run the clustering algorithm $cal(C)$ on $bold(Q)$, extract all *cluster centroids*  $display(bold(P) = mat(vec(p)^((1)), vec(p)^((2)), dots.c.h, vec(p)^((m))))$.
]

#figure(
  caption: [Clustering method for keypoint detection],
)[
  #image("clustering/keypoint-clustering.svg", width: 40%)
]

== Detecting edges through sampling

For each pair of keypoints, we want to establish a graph of connecting edges between the keypoints if there is a line segment connecting them on the image. As the segmented mask has seperated the filed lines with the rest of the image, yet still noisy, we need a robust algorithm to detect an edge between any keypoints pair.

Our idea is to sample as many points as possible lying between the two keypoints $vec(p)_1$ and $vec(p)_2$. Such sampled points will take the form $vec(q) = (1-t) vec(p)_1 + t vec(p)_2$, where $t$ is a parameter within the $[0,1]$ range. For additional robustness, an error term $vec(bold(epsilon))$ is added to the sampled point $vec(q)$. This prevents us from misdetecting edges whose endpoints lie at the border of the edge (as all edges in this case have thickness, therefore it matters where the keypoints are located). The number of sampled points, $T$, can be taken in proportion with the length of the segment $bold(p)_(12) = (vec(p)_1, vec(p)_2)$.



#figure(
  caption: [Clustering method for keypoint detection],
)[
  #grid(
    columns: (63%, 36%),
    image("clustering/edge-sampling.svg", width: 75%), image("clustering/gamma-correction.png"),
  )
]



#colored-quote(fill: yellow.transparentize(66%))[
  *Edge detection via sampling*.
  - _Input_.
    - Edge mask $I_E$ of size $H times W$;
    - Two keypoints $vec(p)_1$ and $vec(p)_2$;
    - $gamma approx 0.01$
  - _Output_. Confidence if there is an edge connecting $vec(p)_1$ and $vec(p)_2$, between $0$ and $1$.

  Step 1. Let $T := ceil(||vec(p)_1 - vec(p)_2||)$.

  Step 2. Generate $T$ random values of $t^((i)) in [0, 1]$.

  Step 3. For each $t^((i))$, the sampled point is $vec(q)^((i)) := (1 - t^((i))) vec(p)_1 + t^((i)) vec(p)_2 + vec(bold(epsilon))$, where $vec(bold(epsilon))$ is a random noise vector, taking range $[-2, 2]$ for each dimension.

  Step 4. Return confidence as the sum of the points' intensities raised to the power of $gamma$:
  $ "Confidence" := sum_i I_E (q^((i))_x, q^((i))_y)^gamma $
]

== Detecting groups of orthogonal lines via clustering of gradients

#figure(
  caption: [Classified segments based on the gradient vector \ _(red for vertical field lines, green for horizontal field lines, and yellow for upright)_],
)[
  #image("clustering/gradients-clustering.svg", width: 130%)
]

#colored-quote(fill: yellow.transparentize(66%))[
  *Grouping lines by the gradient vector*.

  _Input_.
  - Set of lines $L = {hvec(l)^((i))}$

  _Output_. Two sets of parallel lines $L_1$ and $L_2$, where each line in $L_1$ is orthogonal with a line in $L_2$ in the original space.

  Step 1. For each line $hvec(l)^((i))$, compute the normalised gradient vector $display(unit(v)^((i)) = 1/(sqrt(l_1^(i)^2 + l_2^(i)^2)) mat(l_1^((i)); l_2^((i))))$

  Step 2. Use a clustering algorithm $cal(C)$ to group the lines into two groups.
]

#figure(
  caption: [Classified segments based on the gradient vector \ _(red for vertical field lines, green for horizontal field lines, and yellow for upright)_],
)[
  #image("clustering/edge-detected.png", width: 90%)
]


= Homographic rectification

For theoretical and computational methods regarding the homographic transformation, the report refers to Hartley and Zisserman @hartley_multiple_2003[Chapter 2] on _Projective geometry and Transformations of 2D_.

== Projective geometry

Consider the $RR^3$ space with the plane $(pi): z = 1$, which we denote as the *projective plane*. Let $cal(H)$ be a set of points in $RR^3$ (a single point, a line, a shape, etc.) in $RR^3$. Then, for each point $hvec(p)$ in $cal(H)$, the line through $hvec(p)$ and the origin intersects the projective plane $(pi)$ at a point denoted $vec(p)$. _We call $vec(p)$ the projective image of $hvec(p)$ onto the projective plane._

#figure(caption: [])[
  #image("transforms-2d/projective.svg", width: 60%)
]

This projection motivates us to derive an alternative representation for any point $display(vec(p) = mat(x, y)^T)$ on a 2-dimensional (2D) plane by considering it the projective image onto the plane $z = 1$ of any point $hvec(p)$ in $RR^3$. The coordinates of point $vec(p)$ in $RR^3$ is $display(mat(x, y, 1)^T)$ as $vec(p)$ lies on the $z=1$ plane.  Using the argument of similar triangles, one can derive that all possible representations of the correspondent finite point $hvec(p)$ is thus $display(k mat(x, y, 1)^T = mat(k x, k y, k)^T)$ for any $k != 0$.

Following this representation, any line $a x+b y+c=0$ on the $RR^2$ plane can also be uniquely defined by a single homogenous vector $display(hvec(l) = mat(a, b, c)^T)$. This gives us the following important, yet computationally neat, corollaries:
- A point $hvec(p)$ is incident with (or lies on) a line $hvec(l)$ iff $hvec(l)^T hvec(p)=0$.
- Two lines $hvec(l)_1$ and $hvec(l)_2$ always intersects at point $hvec(p) = hvec(l)_1 times hvec(l)_2$ (whether the intersection actually corresponds to a real point on the 2D plane is a different problem).
- The line crossing two points $hvec(p)_1$ and $hvec(p)_2$ is computed as $hvec(l) = hvec(p)_1 times hvec(p)_2$.

Note that the second corollary also works for parallel lines, that is, the two lines of form
$display(hvec(l)_1 = mat(a, b, c_1))^T$ and $display(hvec(l)_2 = mat(a, b, c_2))^T$. The intersection of which is,

$
  hvec(l)_1 times hvec(l)_2 = (c_2 - c_1) mat(b; a; 0)
$

Thus, a homogenous vector whose last component is zero can be thought of as the "intersection" of parralel lines, representing "the point at infinity."

// The projective space is also denoted as $PP^2 = RR^3 - {(0, 0, 0)}$.

== Transformations in 2D

A geometric transformation $T$ is a mapping from a shape $cal(H)$ to its image $cal(H')$, preserving certain geometric identities between the original and the transformed shape.

We consider linear transformations in the homogenous coordinates system $PP^2$. These transformations take the form
$
  H: & PP^2    & mapsto & PP^2 \
     & hvec(p) & mapsto & H hvec(p) = mat(h_11, h_12, h_13; h_21, h_22, h_23; h_31, h_32, h_33) mat(x_1; x_2; x_3)
$

=== Types of transformations

We consider the following $3$ families of transformation:

// - Translation:
// $
//   hvec(p)
//   = mat(1, 0, v_x; 0, 1, v_y; 0, 0, 1) hvec(p)
//   = T(vec(v)) dot hvec(p)
// $

// - Rotation
// $
//   hvec(p)
//   = mat(cos theta, -sin theta, 0; sin theta, cos theta, 0; 0, 0, 1) hvec(p)
//   = R(theta) dot hvec(p)
// $

- *Similarity*. A similarity is a transformation that preserves angles and length ratios. It can be further decomposed as a chain of a $theta$-rotation about the origin $R(theta)$, followed by a scale $k$ and a translation along the vector $vec(v)$.
#grid(
  columns: (70%, 30%),
  figure()[
    #image("transforms-2d/similarity.svg")
  ],
  $
    hvec(p)' = mat(k cos theta, -k sin theta, v_x; k sin theta, k cos theta, v_y; 0, 0, 1) hvec(p)
  $,
)

- *Affinity*. An affinity is a transformation that preserves parallelisms and length ratios. It can be described as a linear transformation in $RR^2$ where the basis ${unit(i), unit(j)}$ is transformed into ${a_11 unit(i) + a_21 unit(j), a_12 unit(i) + a_22 unit(j)}$. A similarity is also a special case of an affinity where $a_11 = a_22=k cos theta$ and $-a_12 = a_22 = k sin theta$.
#grid(
  columns: (70%, 30%),
  figure()[
    #image("transforms-2d/affinity.svg")
  ],
  $
    hvec(p)' = mat(a_11, a_12, v_x; a_21, a_22, v_y; 0, 0, 1) hvec(p)
  $,
)

- *Homography*. A homography is a transformation that preserves incidence and collinearity, that is, if a point $vec(p)$ belongs to the line $hvec(l)$ (or $hvec(p)^T hvec(l) = 0$), then so is $vec(p')$ on the transformed line $hvec(l)' = H^(-T) hvec(l)$ @hartley_multiple_2003[p.33]. An affinity is a special case of a homography where $h_31 = h_32 = 0$.
#grid(
  columns: (70%, 30%),
  figure()[
    #image("transforms-2d/homography.svg")
  ],
  $
    hvec(p)' = mat(h_11, h_12, h_13; h_21, h_22, h_23; h_31, h_32, h_33) hvec(p)
  $,
)

In field registraion we are interested in recovering the homography matrix in two conditions:
- where all the points, pre-transformed ${hvec(p)^((i))}$ and post-transformed ${hvec(q)^((i))}$, and their correspondences $hvec(p)^((i)) <-> hvec(q)^((i))$ are well established _(The DLT algorithm)_; or
- where the pre-transformed $P = {hvec(p)^((i))}$ and post-transformed $Q = {hvec(q)^((i))}$ points are not fully known, and no correspondences have been found. _(Rectification + RANSAC)_.


// $
//   hvec(l)' &= H hvec(p)^((1)) times H hvec(p)^((2)) \
//            &= mat(
//               hvec(h)_(1)^T hvec(p)^((1));
//               hvec(h)_(2)^T hvec(p)^((1));
//               hvec(h)_(3)^T hvec(p)^((1))
//             ) times
//             mat(
//               hvec(h)_(1)^T hvec(p)^((2));
//               hvec(h)_(2)^T hvec(p)^((2));
//               hvec(h)_(3)^T hvec(p)^((2))
//             ) \
//             &= mat(delim: "|",
//               unit(i), hvec(h)_(1)^T hvec(p)^((1)), hvec(h)_(1)^T hvec(p)^((2));
//               unit(j), hvec(h)_(2)^T hvec(p)^((1)), hvec(h)_(2)^T hvec(p)^((2));
//               unit(k), hvec(h)_(3)^T hvec(p)^((1)),  hvec(h)_(3)^T hvec(p)^((2))
//             ) \
//             &= mat(
//               hvec(p)^(1)^T (hvec(h)_2 hvec(h)_3^T - hvec(h)_3 hvec(h)_2^T) hvec(p)^((2));
//               hvec(p)^(1)^T (hvec(h)_3 hvec(h)_1^T - hvec(h)_1 hvec(h)_3^T) hvec(p)^((2));
//               hvec(p)^(1)^T (hvec(h)_1 hvec(h)_2^T - hvec(h)_2 hvec(h)_1^T) hvec(p)^((2))
//             )
//             &= hvec(p)^(1)^T () hvec(p)^((2))
// $


== Solving linear systems using the Singular Value Decomposition (SVD)

Often when solving for parameters of a certain geometric entity or transformation, it is needed to solve for the non-trivial roots of the correspondent linear system, for instance, $A vec(x) = vec(0)$. However, due to noise and errors arising from preceding calculations, the system can either be underdetermined or overdetermined.

A common workaround is to frame the problem as a minimisation of the norm of the residual vector $norm(A unit(x))^2$, with the restriction $||unit(x)||^2=1$ to avoid the trivial solution $unit(x) = bold(0)$. One common method for these types of problem is the Singular Value Decomposition (SVD) algorithm.

From the SVD, any matrix $A_(m times n)$ can be written as
$
  A = U D V^T = sum_(i=1)^n sigma_i unit(u)_i unit(v)_i^T
$

where $sigma_1 >= sigma_2 >= dots.c.h >= sigma_r > 0$ are the singular values of $A$, and $U = display(mat(unit(u)_1, unit(u)_2, dots.c.h, unit(u)_r))$, and $V = display(mat(unit(v)_1, unit(v)_2, dots.c.h, unit(v)_r))$ satisfying $U^T U = V^T V = I$.

This representation gives us a straightforward solution: $display(arg min_(unit(x): ||unit(x)||=1) ||A unit(x)||^2 = unit(v)_r)$, i.e. the right vector associated with the least non-zero singular value. A short proof is given in @hartley_multiple_2003[A5.3].  The SVD sets the foundations for solving linear systems describing homographic transformations under imperfect conditions.

== Recovering homography using point-to-point correspondence (Direct Linear Transform, DLT)

Given $k$ point-to-point correspondences $vec(p)^((i)) <-> vec(q)^((i))$, where $k >= 4$, find a homographic transformation $H$ that maps every point $vec(p)^((i))$ to $vec(q)^((i))$.  Note that each correspondence $vec(p)^((i)) <-> vec(q)^((i))$ provides the equation $H hvec(p)^((i)) tilde hvec(q)^((i))$, or:
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


Note that the left matrix $A^((i))$ has rank $2$, therefore we only need to keep two of the three rows. As a homographic transformation $H$ has $8$ DoF, a minimum of $4$ correspondences is required for the system to be determined.

#colored-quote(
  fill: red.transparentize(70%),
)[
  *The Direct Linear Transform (DLT) algorithm*.

  _Input_. $k$ correspondences $p^((i)) mapsto q^((i))$ under the homographic transformation $H$. \
  _Output_. Best-fitting homography matrix $H$.

  Step 1. For each correspondence $hvec(p)^((i)) <-> hvec(q)^((i))$, prepare the matrix
  $
    A^((i)) = mat(
      - p^((i))_1 q^((i))_3, - p^((i))_2 q^((i))_3, - p^((i))_3 q^((i))_3, , , , p^((i))_1 q^((i))_1, p^((i))_2 q^((i))_1, p^((i))_3 q^((i))_1;
      , , , p^((i))_1 q^((i))_3, p^((i))_2 q^((i))_3, p^((i))_3 q^((i))_3, - p^((i))_1 q^((i))_2, - p^((i))_2 q^((i))_2, - p^((i))_3 q^((i))_2;
    )
  $

  Step 2. Stack all the matrices $A^((i))$ into a single matrix $A$ of size $2k times 9$:
  $
    A = mat(A^((1)); A^((2)); dots.v; A^((k)))
  $
  Step 3. Solve for $display(hvec(h) = arg min_(||hvec(h)'||=1) ||A hvec(h)'||^2)$ via the SVD. _If $k = 4$, the linear system is consistent_.

  Step 4. Reshape $hvec(h)$ from size $9 times 1$ to the matrix $H in RR^(3 times 3)$.
]

== Rectification

Given two groups of parallel lines $display(L = mat(dots.c.h, hvec(l)^((i)), dots.c.h))$ and $display(M = mat(dots.c.h, hvec(m)^((j)), dots.c.h))$ such that each pair of lines $(hvec(l)^((i)), hvec(m)^((j)))$ is orthogonal. Under the homography $H$, the groups become $display(L' = H L = mat(dots.c.h, hvec(l)'^((i)), dots.c.h))$ and $display(M' = H M = mat(dots.c.h, hvec(m)'^((j)), dots.c.h))$. Find the best-fitting homography $H$.

We can approach this problem in a step-by-step manner, implying each additional constraint at each step:

- Firstly, to restore affinity, that is, to apply a "pure" homography $ display(H_P = mat(1; , 1; v_x, v_y, 1)) $ such that if $L' mapsto^(H_P) L_P$ and $M mapsto^(H_P) M_P$, then each line in $L_P$ is pairwise parallel, and each line in $M_P$ is pairwise parallel. The process is called *affine rectification*.

- Secondly, to restore similarity, that is, to apply a "pure" affinity $ display(H_A = mat(a_11, a_12; a_21, a_22; , , 1)) $ such that if $L_P mapsto^(H_A) L_A$ and $M_P mapsto^(H_P) M_A$, then the transformed image is only different from the original one by a similarity. This process is called *metric recrification*.

- Finally, to restore the original coordinates of the lines, that is, to apply a similarity $ display(H_S = mat(k cos theta, - k sin theta, t_x; k sin theta, k cos theta, t_y; , , 1)) $ such that if $L_A mapsto^(H_S) L_S$ and $M_A mapsto^(H_P) M_S$, then $L_S = L$ and $M_S = M$, that is the original lines are fully recovered.

The rectification homography is thus $H' = H_P H_A H_S$, therefore the homography from the original image to the transformed image is the inverse, $H = H'^(-1) = H_S^(-1) H_A^(-1) H_P^(-1)$.


=== Affine rectification

The two vanishing points $hvec(p)_l^((infinity))$ and $hvec(p)_m^((infinity))$ for the $ell$-direction and $m$-direction can be determined by solving the linear system: $L^T hvec(x) = hvec(0)$ and $M^T hvec(x) = hvec(0)$. This can be done via the SVD. The vanishing line is therefore $hvec(v)^((infinity)) = hvec(p)_l^((infinity)) times hvec(p)_m^((infinity))$.

The homography for affine rectification is thus, $display(H_P = mat(1; , 1; v_x^((infinity)), v_y^((infinity)), 1))$.

=== Metric rectification

For metric rectification, we either need a pair of vanishing points and a pair of orthogonal lines, or five pairs of orthogonal lines @hartley_multiple_2003[p.57]. It is clear that for our application, the former option is more optimal.

Consider each pair of lines in the affinely rectified space $(hvec(l)_A^((i)), hvec(m)_A^((j)))$. Note that the first two components of the original lines $hvec(l)^((i)) = (l_1^((i)), l_2^((i)), l_3^((i)))$ and $hvec(m)^((j)) = (m_1^((j)), m_2^((j)), m_3^((j)))$ are the slope vector of the respective lines. Because the original lines are orthogonal, $l_1^((i))m_1^((j)) + l_2^((i))m_2^((j))=0$, or in matrix form,

$
  hvec(l)^(i)^T mat(1; , 1; , , 0) hvec(m)^((j)) = 0
$

An affinity $display(H_A = mat(bold(K), ; , 1))$ transforms the lines $hvec(l)_A^((i)), hvec(m)_A^((j))$ to $H_A^(-T) hvec(l)_A^((i)), H_A^(-T) hvec(m)_A^((j))$ respectively @hartley_multiple_2003[p.33].
$
          & (H_A^(-T) hvec(l)_A^((i)))^T mat(1; , 1; , , 0) (H_A^(-T) hvec(m)_A^((j))) = 0 \
  => quad & hvec(l)_A^(i)^T (H_A mat(1; , 1; , , 0) H_A^T) hvec(m)_A^((j)) = 0 \
  => quad & hvec(l)_A^(i)^T mat(bold(K); , 1) mat(bold(I); , 0) mat(bold(K)^T; , 1) hvec(m)_A^((j)) = 0 \
  => quad & hvec(l)_A^(i)^T mat(bold(K) bold(K)^T; , 0) hvec(m)_A^((j)) = 0 \
$

=== Similarity rectification

#figure(
  caption: [Transformation $H_P H_A$ from original segments (red) to metrically rectified segments (blue) (left); Metrically rectified segments (blue) and the target field (green)],
)[
  #image("rectification/similarity-rectification.png", width: 100%)
]

The task at this stage is to find a similarity $H_S$ to map a set of keypoints and segments $bold(P)$ and the original field $bold(Q)$, with the notice that
- Only a small subset of keypoints and segments in $bold(Q)$ has a correspondence in $bold(P)$;
- No direct correspondence between segments or keypoints are known; and
- A segment $bold(p)^((i j)) := (vec(p)^((i)), vec(p)^((j)))$ may only map to a portion of the segment $bold(q)^((i j)) := (vec(q)^((i)), vec(q)^((j)))$. Nonetheless, there is a high chance that there exists a segment in $bold(P)$ that perfectly maps onto $bold(Q)$.

Moreover, we notice that a similarity has $4$ degrees of freedom. Indeed, let $(a,b,c,d) := (k cos theta, k sin theta, t_x, t_y)$, the similarity takes the form $display(H_S = mat(a, -b, c; b, a, d; , , 1))$. Therefore, it is sufficient to indicate a segment at pre-transformation $(vec(p)^((i)), vec(p)^((j)))$ and post-transformation $(vec(q)^((i)), vec(q)^((j)))$.

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

The final task is to actually find such a correspondence of segments. With non-labelled segments, the only method is to bruteforce through each pair of segments. The question is how to derive a metric to evaluate if a correspondence is "appropriate."

We borrow some ideas from the "RANdom SAmple Consensus" (RANSAC) @hartley_multiple_2003[p.117, Section 4.7.1] to propose an evaluation metric of a segment correspondence, using the pairwise "count of inliers" metric. Suppose after the similarity transformation we find the correspondences $bold(p)^((a_i b_i)) <-> bold(q)^((c_i, d_i))$. Then we want to the maximise the number of segments in $P$ having a $Q$-correspondence, and vice versa. Formally, denote $S_P$ be the set of segments in $P$ whose $Q$-correspondence exists, and similarity denote $S_Q$. Then the quantity $|S_P| + |S_Q|$ represents the "inliers" and is desired to be maximised.

#figure()[
  #image("rectification/inlier.svg", width: 60%)
]

#colored-quote(fill: red.transparentize(66%))[
  *Modified RANSAC to find segment correspondence*.

  // - _Input_:
  //   - All segments
  //   - A threshold $T$
  // - _Output_:

  _Step 1_. Let $S_P := emptyset$ and $S_Q := emptyset$

  _Step 2_. Select a random segment $bold(p)^((alpha beta)) = (vec(p)^((alpha)), vec(p)^((beta)))$ in $bold(P)$ and a random segment $bold(q)^((gamma delta)) = (vec(q)^((gamma)), vec(q)^((delta)))$.

  _Step 3_. Find a similarity $H_S$ such that the segment $bold(p)^((alpha beta))$ is mapped to $bold(q)^((gamma delta))$.

  _Step 4_. For each segment $bold(p)^((a b))$ in $bold(P)$ and $bold(q)^((c d))$

  - Calculate the distance between each point in segment $bold(p)^((a b))$ to the line containing $bold(q)^((c d))$:
  $
    d_a := op("dist")(vec(p)^((a)), bold(q)^((c d))) \
    d_b := op("dist")(vec(p)^((b)), bold(q)^((c d)))
  $

  - Let $display(d_"avg" = 1/2 (d_a + d_b))$. If $d_"avg" <= T$, $S_P := S_P union {a b}, S_Q := S_Q union {c d}$.

  _Step 5_. Let $"Score" = |S_P| + |S_Q|$. If $"Score" >= "Score"_"Accept"$, return $H_S$. Otherwise, repeat the process.
]


== Camera calibration

$ P = K R mat(I_3 | vec(t)) $

= Experiments



== Dataset Preparation

The SoccerNet Calibration Dataset (`sn_calibration`) @noauthor_soccernet_2025 consists of $12356$ training samples, $2796$ validation samples and $2719$ testing samples. Our actual training and validation processes only use a subset of the given samples for model prototyping purposes.

#figure(
  caption: [],
)[
  #table(
    columns: (150pt, 75pt, 75pt, 75pt),
    table.header([*Task*], [*Training*], [*Validation*], [*Testing*]),
    [_General_], [$12356$], [$2796$], [$2719$],
    [_Edge segmentation_], [First $128$], [First $64$], [First $64$],
    [_Keypoint concentration_], [First $128$], [First $128$], [First $128$],
  )
]

Each sample contains two files: a `PNG` colour image of size $960 times 540$px and a `JSON` data file containing the coordinates of $26$ different field lines and curves. For this project, we ignore the three curves (the 10-yard kick-off circle and the two semicircles). Each of the lines is described by a segment (or a poly-segment) of two or more keypoints.

Assuming every football field has the same size and layout #footnote[This is not always the case. The current regulation allows for fields to have a variable size of 90-120m in length and 50-90m in width, therefore each field may have a different size than another. Our project uses the field size of $110 times 75$m.] of $110 times 75$m, we can always map a template football field to the actual field displayed on the image via a homography $H$. Such homography can be reconstructed using the DLT algorithm when more than $4$ point-to-point correspondences are found.

For instance, given the coordinates of the field's top and left border lines, the top left corner keypoint can be computed as:
$ hvec(p)_"FIELD_TOP_LEFT" = hvec(l)_"FIELD_BOX_TOP" times hvec(l)_"FIELD_BOX_LEFT" $

This can be done for all the visible keypoints on the field, as well as off-screen keypoints whose constituent lines are visible. Using the found correspondences, apply the DLT algorithm directly to find the homography $H$. All other keypoints can be found by applying the homography on the template field.


#figure(
  caption: [Defined field lines in the dataset @noauthor_soccernet_2025 (left); Keypoint recovery method (right)],
)[
  #grid(
    columns: (50%, 50%),
    gutter: 10pt,
    image("prep/annotations.png"), image("prep/annotation-fill.svg"),
  )
]

Given that at least $4$ non-collinear keypoints are positioned, the coordinates of the other of the $27$ keypoints can be detected via a homography transformation. This can be done via the DLT algorithm.

The dataset is then catered to different types of problem via `pytorch`'s `Dataset`s. For efficient batching, the labels must be in a `Tensor` form for stacking and parallel computation. Therefore, we implemented a `Dataset` interface for every use case in the pipeline, each reading from the same original dataset from SoccerNet @noauthor_soccernet_2025, but with additional preprocessing (such as grayscaling, homography reconstruction and transformation, etc.).

#figure(
  caption: [],
)[
  #table(
    columns: (42%, 21%, 40%),
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
]



== Image segmentation

#figure(
  caption: [],
)[
  #image("unet/unet-seg-metric.png")
]

#figure(
  caption: [],
)[
  #image("unet/unet-kp-metric.png")
]


== Field registration via homography transformation


= Discussion


#bibliography("refs.bib")
