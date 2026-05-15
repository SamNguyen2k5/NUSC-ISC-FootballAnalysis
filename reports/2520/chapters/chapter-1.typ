= Introduction

The previous Independent Study Course (ISC-1) established a three-step machine learning pipeline method for the field localisation task, i.e. to determine the region and boundaries of a football field from single-view frames. However, during the actual development stage, many technical difficulties were soon realised, steering the project away from original methodological research to repetitive pipeline engineering tasks. Acknowledging the challenges mentioned above, this project aims to evaluate the efficiency of pipeline automation measures in the context of single-frame football field localisation.

Specifically, the proposed benefit for this ISC project (ISC-2) is two-fold:

- To automate certain parts of the development of the existing project. The project shall analyse the repetitive tasks observed in the process of training and deploying ML models, then automate them through an ML library. This process is expected to speed up the development process and utilise computational resources economically.

- To adapt the current project for participation in a new official machine learning challenge. This report introduces a new field localisation task, Spiideo SoccerNet SynLoc 2026 @ardo_spiideo_2025, a live challenge introduced by SoccerNet 2026. This new task is a continuation of previous tasks on field localisation and camera calibration in SoccerNet 2025, with two main differences of the input feed:

  - _(a) they are single- and wide-framed, covering a majority of the field; and_
  - _(b) they are subject to radial distortion._

// In the previous ISC, we concluded that the proposed three-step pipeline works best for broad-view images; therefore, there is evidence that we can adapt such an approach for this particular task, after performing self-calibration to remove any non-affine distortion.

#figure(caption: [
  Image from the Camera Calibration @falaleev_enhancing_2024 task (left) versus the Spiddeo SymLoc dataset (right)
])[
  #grid(
    columns: 2,
    column-gutter: 3pt,
    image("../dataset/desc-calibration.jpg"), image("../dataset/desc-spiideo.jpg"),
  )
]

This report serves as a reflection after establishing the needed infrastructure for scaling up the project, and the project's initial participartion in the Spiideo SynLoc challenge @ardo_spiideo_2025, with qualitative methodology reviews and actionable future plans moving forward.
