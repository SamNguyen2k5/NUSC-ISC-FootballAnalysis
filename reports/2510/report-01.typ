#import "@preview/bamdone-ieeeconf:0.1.1": ieee

#set page(paper: "a4", margin: 2cm)
#set text(size: 12pt)
#set par(leading: 0.6em, justify: true)
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
#show: ieee.with(
  title: [
    Theoretical Foundations for 3D Reconstruction of Sports Games with Limited Visual Data Settings
  ],
  abstract: [
    This electronic document is a live template. The various components of your paper [title, text, heads, etc.] are already defined on the style sheet, as illustrated by the portions given in this document.
  ],
  authors: (
    (
      given: "Nguyen Hoang The Kiet",
      surname: "",
      email: [albert.author],
      affiliation: 1,
    ),
    // (
    //   given: "Bernard D.",
    //   surname: "Researcher",
    //   email: [b.d.researcher],
    //   affiliation: 2
    // )
  ),
  // affiliations: (
  //   (
  //     name: [Faculty of Electrical Engineering, Mathematics and Computer Science, University of Twente],
  //     address: [7500 AE Enchede, The Netherlands],
  //     email-suffix: [papercept.net],
  //   ),
  //   (
  //     name: [Department of Electrical Engineering, Wright State University],
  //     address: [Dayton, OH 45435, USA],
  //     email-suffix: [ieee.org]
  //   ),
  // ),
  affiliations: (),
  index-terms: (),
  bibliography: bibliography("refs.bib"),
  draft: true, // Adds the draft markers on the footer and header
  paper-size: "a4",
  disclaimer: "",
)

#set math.mat(delim: "[")
#let vec(v) = { $ harpoon(bold(#v)) $ }
#let hvec(v) = { $ tilde(bold(#v)) $ }
#let unit(v) = { $ hat(bold(#v)) $ }

// Your content goes below.

= Introduction

== Motivation
- VAR controversies (examples at the 2024 ASEAN Cup)
- (The problem of cameras shortage)

== Problem statement

- Input?
- Output?

== Problem analysis

- How many stages are there in a 3D reconstruction pipeline?
  -

= Related Works

== Academic
== Practical

= Field registration

== Image segmentation methods
=== YOLO-based methods
- YOLO segmentation models

=== Generative models
- UNet

== Homography transformation in homogeneous coordinates

=== Homogeneous coordinates
- In 2-dimensional space:
$
  hvec(x) = mat(x; y; t) => vec(x) = mat(x"/"t; y"/"t)
$
- In 3-dimensional space:
$
  hvec(X) = mat(X; Y; Z; T) => vec(X) = mat(X"/"T; Y"/"T; Z"/"T)
$

=== Transformations in homogeneous coordinates

- Translation by a vector
  $vec(v) = mat(display(X_v "/" T_v), display(Y_v "/" T_v), display(Z_v "/" T_v))^T$
$
  vec(X)' = vec(X) + vec(v) =
  mat(X"/"T + X_v"/"T_v; Y"/"T + Y_v"/"T_v; Z"/"T + Z_v"/"T_v)
$

$
  hvec(X)' =
  mat(X"/"T + X_v"/"T_v; Y"/"T + Y_v"/"T_v; Z"/"T + Z_v"/"T_v; 1)
  ~
  mat(
    X T_v + X_v T;
    Y T_v + Y_v T;
    Z T_v + Z_v T;
    T T_v
  )
  \ =
  mat(T_v, , , X_v; , T_v, , Y_v; , , T_v, Z_v; , , , T_v)
  mat(X; Y; Z; T)
$

- Rotation



- Similarity
- Affine

=== The homography transformation
- Homography


- Homography decomposition (https://medium.com/@insight-in-plain-sight/deconstructing-the-homography-matrix-35989ecc0b2)

$
  H & = H_P H_A H_S \
    & =
      mat(1; , 1; v_x, v_y, 1)
      mat(a_11, a_12; a_21, a_22; , , 1)
      mat(cos theta, -sin theta; sin theta, cos theta; , , 1"/"s)
$


=== The 4-point algorithm

$$


== Camera calibration

$ P = K R mat(I_3 | vec(t)) $

= Experiments
== Image segmentation

== Field registration via homography transformation


= Discussion
