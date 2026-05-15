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
  ISC 2,  Checkpoint 2
]
== Ideas

=== Models

#figure()[
  #table(
    align: (left, left, center),
    columns: (50%, auto, auto),
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
      - Train a *YOLO* detector to detect regions associating to each player
      - Determine the keypoints for each player based on the cropped region
    ],
    [
      - Images are blurry and contain artifacts $-->$ harder to detect objects

      - #text(fill: red)[
          (Ultralytics) YOLO does not provide a flexible interface for training (no custom dataset, etc.)
        ]
    ],
    [
      #text(fill: red, weight: "bold")[-/-]
    ],
    [
      - Segmentation model (*UNet*-based)
      - Determine the keypoints for each player based on the cropped region
      - Code: `./models/base/unet_heatmap.py`
    ],
    [
      - UNet isn't stable, may produce false negative pixels.
      - This behaviour can be seen from the `MaskCalibration` model.
    ],
    [
      #text(fill: orange, weight: "bold")[in research]
    ],
    [
      - *Gaussian Mixture Model* (?)
      - Treat each player as a Gaussian distribution (heatmap, etc.)
      - Code: `./pipelines/player_detection/location_detector.py`
    ],
    table.cell(
      align: center,
    )[-/-],
    [
      #text(fill: orange, weight: "bold")[in research]
    ],
  )
]

=== Dataset preparation (`./datamodules/spiideo`)

_During training:_

- From the original image and annotation, generate a Gaussian heatmap, where each player is a Gaussian heat source with variance $sigma$. The heatmap idea for detecting small objects is described in @liu2024esodefficientsmallobject.

- Select $k dot n_"crop"$ random crops, each of size $w times w$.
- Sort the image and mask crops in order of decreasing "intensity". For simplicity we use the sum of pixel intensities in the target mask.

- Choose $r dot n_"crop"$ with the highest "intensity" and $(1 - r) dot n_"crop"$ with the lowest "intensity."

- Hyperparameters: $sigma=32,16,8$; $k = 2,3,5$; $r=95\%; n_"crop"=16$

#figure()[
  #image("dataset/gaussian_masks_preprocess.png")
]

#figure(
  caption: [
    Top: batched cropped images; Bottom: Gaussian masks. \
  ],
)[
  #image("dataset/gaussian_mask_example.png")
]

_During inference_:
- From the original image, divide the image into smaller chunks, each of size $w times w$.

- For each chunk $I^((i, j))$:
  - Generate the Gaussian heatmap using the trained model $H^((i, j)) := cal(H)(I^((i, j)))$.

  - Use the Gaussian Mixture Model approach to detect the centers of the sources in the output mask.


=== Loss functions (`./losses`)
#figure()[
  #table(
    align: (left, left, center),
    columns: (50%, auto, auto),
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
      - Dice Loss
      - A measure of overlap between the ground truth and the prediction mask
      - Has been successful for the field lines segmentation model (ISC-1)
      - Hard DiceLoss:
      $ "DiceLoss" = 1 - (2|Y inter hat(Y)|)/(|Y| + |hat(Y)|) $
      - Soft DiceLoss: (for masks with sub-unit pixels)
      $
        "DiceLoss" = 1 - sum_t frac(
          bold(y)_t bold(hat(y))_t,
          bold(y)_t dot bold(1) +
          bold(hat(y))_t dot bold(1)
        )
      $
    ],
    [
      - If $Y$ and $hat(Y)$ are non-overlapping masks, training on DiceLoss is impossible as $hat(Y)$ then becomes a *local minimum*.
    ],
    [
      #text(fill: orange, weight: "bold")[in research]
    ],

    [
      - Focal Loss
      - Derived from the Binary Cross-Entropy Loss
      - Heavier penalty on False Positives
      $ "FL" = -sum_t alpha_t y_t (1-hat(y)_t)^gamma log hat(y)_t $
    ],
    [

    ],
    [
      #text(fill: orange, weight: "bold")[in research]
    ],

    [
      - Total Variation Loss
      - To reduce noise in the output mask
      $ "TV" = 1/(n) sum_(i,j){ |hat(y)^((i,j)) - hat(y)^((i+1,j))| + |hat(y)^((i,j)) - hat(y)^((i,j+1))| } $
    ],
    [
      - Incentivises the model to output a blank black mask so that $"TV"=0$.
    ],
    [
      #text(fill: orange, weight: "bold")[in research]
    ],
  )
]

== Experiments
See the #link("https://wandb.ai/samau07012005-national-university-of-singapore-students-/ISC-Football/reports/NUSC-ISC-2-Checkpoint-2--VmlldzoxNjMwMzk3OA?accessToken=4q2j1mlh315kithzmcjk44lk6f5b32ginx9xl1fpkq7hk4qy6rvq2ywfdqo9lexc")[#underline()[wandb report here.]]



== Future plans

=== Considerations

- Is the task definition (constructing a model that transforms an image $->$ a heatmap with Gaussian sources) appropriate as a learnable task? What are the possible alternative data augmentation methods that could be considered?

- What loss functions should be used between the ground truth and the predicted mask, taking note of the high density of black (zero) pixels that may lead the model to stop training after just a few epochs?

- Is U-Net (or other convolutional models) appropriate for the task? Should we also consider other types of models as well (attention-based, transformers, etc.)


=== Timeline

- _14th April_: Official deadline for ISC submission on Canvas.

- _24th April_: Deadline for code submission on Codabench
  - https://www.codabench.org/competitions/10155/#/phases-tab

- _1st May_: Report submission deadline.

=== Administrative

- To negotiate the ISC deadline to closer to the official competition's closing date.

- To consider SoC UROP vs NUSC UROP over the summer

  - https://www.comp.nus.edu.sg/programmes/ug/project/urop/forms/

#bibliography("refs.bib")
