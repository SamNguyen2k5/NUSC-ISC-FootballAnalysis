// #import "@preview/bamdone-ieeeconf:0.1.1": ieee
#import "@preview/basic-report:0.3.1": *
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node

#set page(
  paper: "a4",
  margin: 2cm,
  background: rotate(24deg, text(15pt, fill: rgb("#ff3f2145"))[
    // * DRAFT * \
    // * DRAFT * \
    // * DRAFT *
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

// #outline(
//   title: [List of Figures],
//   target: figure.where(kind: image),
// )

= Theoretical Foundations for 3D Reconstruction of Sports Games with Limited Visual Data Settings

The 2022 FIFA World Cup was the first major football tournaments to introduce a semi-automated offside technology (SAOT) to reconstruct the 3D scene of the players and the ball on a football pitch when an offside call is made. Despite being a major breakthrough in football refereeing and broadcasting, this technology remains inaccessible to many less developed countries and regional tournaments. This ISC project aims to recreate such a technology in the context of low camera coverage and low quality data. To achieve that, this project looks into the theoretical foundations into building a minimum-viable-product (MVP) styled three-step pipeline that is able to reconstruct the full football field model, from segmentation of field lines, reconstructing the graph skeleton representing the field, to recovering the homography transformation from a template field to the target field.
