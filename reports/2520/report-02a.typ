#set page(
  paper: "a4",
  // header: [_INDEPENDENT STUDY COURSE_],
  footer: [#h(1fr) _Nguyen Hoang The Kiet_],
  margin: 1.5cm,
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
  inset: 4pt,
  radius: 4pt,
  spacing: 10pt,
)

#let colored-quote(
  fill: luma(243),
  body,
) = block(
  width: 100%,
  fill: fill,
  inset: 10pt,
  radius: 4pt,
  spacing: 10pt,
)[
  #body
]

#title()[
  ISC 2,  Checkpoint 1
]

= Infrastructure

#image("infra/infra.drawio.png")

#colored-quote()[
  - ⚠️ Wandb account is currently on Trial and will expire soon.
  - Considerations to apply for Wandb for Academics.
]

#pagebreak()

== Task Description

- https://github.com/Spiideo/sskit
- https://www.scitepress.org/publishedPapers/2025/131082/pdf/index.html

- Dataset includes $40$k+ training / $~7$k validation/ $~9$k test images of football matches, synthetically generated from 3D rendering techniques.
  - COCO format: `image`, `annotations`, `categories`.

- Features per image:
  - Calibration parameters:
    - _Camera matrix_: $bold(P)_(3 times 4) = [bold(R)_(3 times 3) | bold(tilde(t))]$
    - _Industrial distortion model_:
      - Normalised polar coordinates:
        - $(u, v) = display(1/w (x-w/2,y-h/2))$
        - $r = sqrt(u^2 + v^2)$
      - Distortion: $r_"dist" = p_"dist" (tan^(-1) (r_"undist"))$
      - Undistortion: $r_"undist" = tan(p_"undist" (r_"dist"))$
      - $p_"dist"$ and $p_"undist"$ are given polynomials. $p_"dist"^(-1) approx p_"undist"$

- Features per each annotated player:
  - Keypoints
    - 2 keypoints recorded, at the pelvis and at its projection to the ground.
    - Both the world coordinates (3D, in world meters) and the image coordinates (2D, in image pixels) are provided.

  - *Location on the pitch* (in meters)
  - Bounding box

- Task: *determine the pitch location of every player*.

#figure(
  caption: "Spiideo synthetic dataset. Each player is annotated by two keypoints: the pelvis and the projection of the pelvis onto the field ground.",
)[
  #image("dataset/kp.png", width: 80%)
]

#figure(
  caption: "Spiideo synthetic dataset, after undistortion.",
)[
  #image("dataset/kp-undistorted.png", width: 80%)
]

- Metric: _mAP-LocSim_
  - $"LocSym"(bold(x)_1, bold(x)_2) = display(exp(ln 0.05 dot d(bold(x)_1, bold(x)_2)^2/tau^2))$, where $tau= 1 "m"$ is the tolerance constant --> "target on precise location"

== Ideas

#figure()[
  #table(
    align: (left, left, center),
    columns: 3,
    table.header(
      table.cell(
        align: center,
        [*Approach*],
      ),
      table.cell(
        align: center,
        [*Drawbacks*],
      ),
      table.cell(
        align: center,
        [*Status*],
      ),
    ),
    [
      - Train a YOLO detector to detect regions associating to each player
      - Determine the keypoints for each player based on the cropped region
    ],
    [
      - Images are blurry and contain artifacts $-->$ harder to detect objects
    ],
    [
      #text(fill: orange, weight: "bold")[experiment]
    ],
    [
      - Segmentation model (UNet-based)
      - Determine the keypoints for each player based on the cropped region
    ],
    [
      - UNet isn't stable, may produce false negative pixels. This behaviour can be seen from the `MaskCalibration` model.
    ],
    [
      #text(fill: black, weight: "bold")[in research]
    ],
    [
      - Gaussian Mixture Model (?)
      - Treat each player as a Gaussian distribution (heatmap, etc.)
    ],
    table.cell(
      align: center,
    )[-/-],
    [
      #text(fill: black, weight: "bold")[in research]
    ],
  )
]
