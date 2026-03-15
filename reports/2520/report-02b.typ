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

= YOLO

== References
- https://www.v7labs.com/blog/yolo-object-detection

- https://www.datacamp.com/blog/yolo-object-detection-explained


== Conversation with Gemini

https://notebooklm.google.com/notebook/babc6073-543f-44f5-bc59-249a43e622b8

== Problem specification:

==== Input
- Image: $X in RR^(C times H times W)$
- List of possible classes $cal(C) = {c_1,c_2,dots,c_m}$

==== Output
- Bounding boxes

=== 1. Grid division
- Divide the image into a grid $S times S$. Box at row $i$ column $j$ denoted as $"Cell"^((i j))$.

=== 2. Bounding box regression
- For each grid cell, calculate the bounding box within the cell

$
  "Cell"^((i j)) -->^(f_1) [bold(B)^((i j, k)) | bold(C)^((i j, k))] =
  mat(
    p^((k)), b^((k))_(x), b^((k))_(y), b^((k))_(w), b^((k))_(h)
    | c^((k))_1, c^((k))_2, dots, c^((k))_(m)
  )_(B times (5 + m))
$

#figure()[
  #table(
    columns: 3,
    align: (center, left, center),
    [*Variable*], [_Description_], [_Range_],
    [$p$], [probability/confidence of $"Cell"^((i j))$ containing an object], [$(0, 1)$],
    [$b_([.])$],
    [
      relative centre location and dimensions of the bounding box \

      - $"actual "x = sigma(b_x) + g_x$
      - $"actual "y = sigma(b_y) + g_y$
      - $"actual "w = g_w e^(b_w)$
      - $"actual "h = g_h e^(b_h)$
    ],
    [$(0, 1)$],

    [$c_j$], [$PP["class" c_j | "object"]$], [$(0, 1)$],
  )
]

=== 3. Non-Maximum Supression


- Sort bounding boxes by confidence, select top $k$

- Select based on an IoU threshold


=== A. Design decisions

- Coupled vs decoupled head.


== Ultralytics's YOLO `Results` object
- https://docs.ultralytics.com/reference/__init__/
- https://docs.ultralytics.com/modes/

- `boxes: Boxes`
- `masks: Masks`



