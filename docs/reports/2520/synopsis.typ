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

== Incremental Development of Machine Learning Systems for Single-Frame Football Field Localisation

This project (ISC-2) continues the work done during the previous Independent Study Course (ISC-1), now adapted to a new task of player localisation. From the challenges facing ISC-1, ISC-2's aim is two-fold. On one hand, this project will attempt to propose a solution to the Spiideo SynLoc Challenge, with the task of player localisation on world coordinates from football wide-view frames. On the other hand, through the development of the project, we shall investigate and apply good practices during project development in the context of data science and machine learning where task definitions, methodological and experimental needs changes frequently.
