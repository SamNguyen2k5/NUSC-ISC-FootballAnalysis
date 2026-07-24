// #import "@preview/bamdone-ieeeconf:0.1.1": ieee
#import "@preview/basic-report:0.3.1": *
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import "@preview/wordometer:0.1.5": total-words, word-count

#show: it => basic-report(
  doc-category: "Independent Study Course (NST3902)",
  doc-title: [Incremental Development of Machine Learning Systems for Single-Frame Football Field Localisation
  ],
  author: "Nguyen Hoang The Kiet",
  affiliation: "NUS College, National University of Singapore",
  // logo: image("assets/aerospace-engineering.png", width: 2cm),
  // <a href="https://www.flaticon.com/free-icons/aerospace" title="aerospace icons">Aerospace icons created by gravisio - Flaticon</a>
  language: "en",
  // compact-mode: true,
  it,
)

#show: word-count

#set page(
  paper: "a4",
  margin: 2cm,
  background: rotate(24deg, text(15pt, fill: rgb("#ff3f2145"))[
    // * DRAFT * \
    // * DRAFT * \
    // * DRAFT *
  ]),

  // footer: [
  //   _Total word count: #total-words words_
  // ],
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

#outline(
  title: [List of Figures],
  target: figure.where(kind: image),
)

#pagebreak()
#include "chapters/chapter-1.typ"
#include "chapters/chapter-2.typ"
#include "chapters/chapter-3.typ"
#include "chapters/chapter-4.typ"
#include "chapters/chapter-5.typ"
#include "chapters/chapter-6.typ"
#include "chapters/chapter-7.typ"

#bibliography("refs.bib")

#set heading(numbering: "A.1.1.", supplement: [Appendix])
#counter(heading).update(0)

#v(15pt)

#line(length: 100%)

= Project Github repository

https://github.com/SamNguyen2k5/NUSC-ISC-FootballAnalysis/tree/isc-checkpoint-2

= Participation in the SynLoc Spiideo Challenge, SoccerNet 2026

- Test server (including submisison from `@samnguyen157`):
  https://www.codabench.org/competitions/10128/#/results-tab

- Challenge server:
  https://www.codabench.org/competitions/10155/#/results-tab

= Live reports on Wandb

https://wandb.ai/samau07012005-national-university-of-singapore-students-/ISC-Football/reports/NUSC-ISC-2-Model-for-competition--VmlldzoxNjcxNzUwNA
