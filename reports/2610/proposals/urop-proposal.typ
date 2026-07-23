// ==========================================
// PREAMBLE: Global Configurations & Styles
// ==========================================

// Default Global Spacing
#let global-spacing = 11pt
#let grid-gap = 11pt
#let table-inset = 6pt
#let line-height = 0.5pt

// Page Setup
#set page(
  paper: "a4",
  margin: (x: 2.5cm, top: 2.5cm, bottom: 2.5cm),
  header: text(fill: luma(100))[
    #grid(
      columns: (1fr, auto),
      [Office of the Provost \ National University of Singapore],
      align(right)[School of Computing \ Updated 11 Jun 2024],
    )
  ],
  footer: align(center)[
    // #text(fill: luma(100))[Page #counter(page).display()]
  ],
)

// Text Setup (Arial, 10pt uniform throughout)
#set text(
  font: "Arial",
  size: 10pt,
  hyphenate: false,
)

// Paragraph & Block Rules
#set par(spacing: global-spacing, leading: 0.65em, justify: true)
#set block(spacing: global-spacing)

// Form Underline Field Helper
#let field(label, content) = {
  block(
    width: 100%,
    stroke: (bottom: line-height + black),
    inset: (bottom: 0.30em),
    [ _#label:_ #content ],
  )
}

// ==========================================
// DOCUMENT BODY
// ==========================================

#align(center)[
  *OFFICE OF UNDERGRADUATE STUDIES* \
  *UNDERGRADUATE RESEARCH OPPORTUNITY PROGRAMME (UROP)* \
  *UROP – Project Proposal Form*
]

// #align(center)[
//   _To be completed for *STUDENT* proposed projects *ONLY*._
// ]

// Students should submit this form to their UROP supervisors, who will in turn put the project information in the Project Admin website at: #link("https://mysoc.nus.edu.sg/~projadm")

// --- Form Fields ---

#field("Project's Title", [

])

#grid(
  columns: (1.2fr, 1fr),
  gutter: grid-gap,
  field("Proposed By", [Nguyen Hoang The Kiet]), field("Student ID No.", [A0305787R]),
)

#grid(
  columns: (1fr, 1fr),
  gutter: grid-gap,
  field("SoC Email", [nhthekiet\@u.nus.edu]), [],
)

#field("Skills", [

])

#v(2em)
*Description of proposed project:*

Modern football needs maximum resolution at the moment of decision-making both on the pitch and on the television screen. This project presents an advanced computer vision framework which allows for the accurate 3D reconstruction of the main players' scene from conventional 2D broadcasts. By focusing exclusively on the key elements of the game, the model isolates and digitises human movements in 3D space which aids both in-crowd refereeing and post-match media analysis.

This project aims to target only the main elements of play. A key moment of the game is broken down into a set of video frames, within which the contours of _relevant players_ are digitally masked using _interactive video segmentation_. This allows for rapid isolation of the figures within the video frames, which are then used to drive the _SMPL model (Skinned Multi-Person Linear)_. The resulting 3D character mesh is then used to render individual characters in a reconstructed 3D space, mimicking their precise movements.

By targeting the key elements of plays requiring review, the project is able to resolve the 3D geometry of the main players involved, providing major benefits for both aspects of football. First, for refereeing, the reconstructed computer vision model can provide millimeter-grade insight into the spatial relations between players and objects to determine whether a foul or offside has occurred for any angle of view. Second, for television broadcasting, 3D reconstruction allows for the creation of targeted VR highlights and replays, providing unlimited viewing angles and opportunities to analyze the context of a call from multiple positions on the pitch.

// --- Signatures ---

#v(6em)
#grid(
  columns: (1fr, 1fr),
  align: (center, center),
  gutter: grid-gap,
  [
    #line(length: 100%, stroke: line-height + black)
    *Name and Signature of STUDENT*
  ],
  [
    #line(length: 100%, stroke: line-height + black)
    *Date*
  ],
)

#v(4em)
#grid(
  columns: (1fr, 1fr),
  align: (center, center),
  gutter: grid-gap,
  [
    #line(length: 100%, stroke: line-height + black)
    *Name and Signature of Supervisor*
  ],
  [
    #line(length: 100%, stroke: line-height + black)
    *Date*
  ],
)

// --- Instructions ---

#text(fill: luma(30%))[
  _
  *Instructions:*
  1. Students are to approach the intended supervisor directly with completed form.
  2. Upon approved, the supervisor will need to proposal the project in the project administration system.
  3. Student are to submit the completed form, with signatures, to Ms Sharifahh at #link("mailto:sha.a@nus.edu.sg")[sha.a\@nus.edu.sg]
  _
]

#pagebreak()

#align(center)[
  *Appendix A. Flowchart of the proposed pipeline*
]

#pagebreak()

#align(center)[
  *Appendix B. Tentative reading list*
]


== Interactive segmentation

- @ravi_sam_2024
- @yao_video_2019

#v(2em)
== SMPL human model

- @loper_smpl_2015
- @bogo_keep_2016
- @shin_multi-view_2020
- @kim_coherent_2026

#v(2em)
#bibliography("refs.bib")
