= Experiments

All experiments are conducted on a Macbook M1 with local M1 MPS chips. The results of training the final model for competition is compiled in
#link("https://wandb.ai/samau07012005-national-university-of-singapore-students-/ISC-Football/reports/NUSC-ISC-2-Model-for-competition--VmlldzoxNjcxNzUwNA?accessToken=ewu4yrxf03tu4criofc95veqnoazzlos3rg9wnphfk0hhm1xa1taib2yu2ka405s", [ #text(fill: blue)[#underline([_this Wandb report_])]]).


#figure(
  caption: [Loss charts after $32$ epochs with a limited dataset of $256$ images, resized at dimensions $720 times 405$.],
  placement: top,
)[
  #image("../experiments/charts.png")
]

Overall, the training results illustrate the model's ability to capture the regions containing the players. The decreasing aggregate training and validation losses after $32$ epochs indicate that the model can be further trained with more epochs and a bigger dataset to achieve better results.

Inspecting the model's outputs gives us a hint on what the model struggles to predict. Two main problems arise through inspection:

- The model struggles to 'find' out players further to the camera than those nearer. Not only does the size of the players on the camera decreases significantly, the players may blend into the background, which either are cropped out by `masking_unet`, or are unable to be detected by `heatmap_unet`.

- The model tends to produce false positives, mostly on the field lines, the intersection of field lines and other foreign objects than players.

#figure(
  caption: [False positive pixels on the boundary between the field area and the spectators' seating area.],
)[
  #image("../experiments/qual_1.png")
]

We also examine our diagnostics by running a full inference pipeline on all `test` and `challenge` images in the Spiideo SynLoc challenge. We take one image from the `test` set as an representative.

- The predictions verify our diagnosed problem during training that the model struggles to predict players further to the camera, whilst predicting false positives when encountering other objects such as field lines or goal posts.

- Some players are annotated twice in the prediction output. This is expected as BGMM (and other mixture-based approaches) fixes a number of components beforehand, and our component weight tolerance is set arbitarily without consulting the dataset.

#figure(
  caption: [Diagnostic of prediction vs ground truth annotations on test image `000977.jpg`],
)[
  #image("../experiments/pred_image.png", width: 100%)
  #image("../experiments/pred_world.png", width: 100%)
]
