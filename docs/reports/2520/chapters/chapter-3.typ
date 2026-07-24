#import "template.typ": *

= Task Description <section:task>

#figure(caption: [
  Input image and keypoint annotations
])[
  #image("../dataset/kp.png")
]

The project has participated in the SoccerNet 2026 compeition on the _"Single Frame World Coordinate AthleteDetection and Localisation_ task @ardo_spiideo_2025. The task consists of two main parts; (a) detecting players on the pitch through a wide-view camera, and (b) localising the players, i.e. locating the world coordinates of each player on the field.

The annotations follow the COCO annotation model @lin_microsoft_2014, where the `JSON` category object only contain a single category `'person'`. Each image has an associated camera matrix $[bold(R) | vec(t)]$, a distortion polynomial $p_"dist"$ and a undistortion polynomial $p_"undist"$. A point $(x,y)$ on the image coordinate system can be transformed into the world coordinate system as follows (see #link("../../../datamodules/spiideo/coco.py", [`datamodules/spiideo/coco.py`])):

- *Undistort the coordinate grid*: $p_"undist"$ is a transformation on the normalised radius component of each pixel. A point $(r,theta)$ in the polar coordinate system is transformed to $(tan(p_"undist" (r)), theta)$ after undisortion. Therefore, let $(u,v)=display(1/w (x-w/2, y-h/2))$ be the normalised cartesian coordinates of the pixel at image coordinate $(x,y)$, then the new radius component of the transformed point is

  $
    r' = tan(p_"undist" (sqrt(u^2+v^2)))
  $

  The new point $(x',y')$ is constructed accordingly.


- *Homographic inverse transformation*: by setting the $z$-coordinate in both the image and the world coorinates to $0$, the third column in the camera matrix can be removed, leaving us with the homographic transformation $bold(H) = display(mat(unit(r)_1, unit(r)_2, vec(t)))$ that maps a world-coordinated point into a image-coordinated pixel. The world coordinates are therefore

$
  lambda mat(x_"world"; y_"world"; 1) = bold(H)^(-1) dot mat(x'; y'; 1)
$

Each player is annotated with two keypoints: one at the point of contact between the player's feet and the field, the other that the player's pelvis. For the purpose of the proposed methodology, we are only interested in the first type of points.


The metric to evaluate the prediction of players' position is the LocSim metric @ardo_spiideo_2025, performed on *world coordinates*, where a ground truth point $vec(G)$ and the prediction point $vec(P)$ have the $"LocSim"$ of
$
  "LocSim"(vec(P), vec(G)) = e^((ln 0.05)||vec(P)-vec(G)||^2)
$

#figure(
  caption: [$"LocSim"$ is performed on world coordinates (right)],
)[
  #grid(
    columns: 3,
    image("../losses/image-coordinates.png"), image("../losses/world-coordinates.png"),
  )
]
