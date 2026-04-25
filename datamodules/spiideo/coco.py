import numpy as np
from typing import List
from pydantic import BaseModel
from numpydantic import NDArray, Shape

class COCOAnnotation(BaseModel):
    id: int
    keypoints: NDArray[Shape['2, 3'], float]                # type: ignore
    keypoints_3d: NDArray[Shape['2, 4'], float]             # type: ignore
    position_on_pitch: NDArray[Shape['2'], float]           # type: ignore
    bbox: NDArray[Shape['4'], float | int]
    area: float
    image_id: int
    category_id: int

class COCOPredictedAnnotation(BaseModel):
    id: int
    area: float = 0
    image_id: int
    category_id: int
    # position_on_pitch: NDArray[Shape['3'], float]           # type: ignore
    position_on_pitch: List[float]
    score: float = 1.0

class COCOImage(BaseModel):
    id: int
    file_name: str
    width: int
    height: int
    camera_matrix: NDArray[Shape['3, 4'], float]            # type: ignore
    dist_poly: NDArray[Shape['9'], float]                   # type: ignore
    undist_poly: NDArray[Shape['9'], float]                 # type: ignore

    def distort(self, u0, v0):
        # u0, v0 = (x0 - self.width / 2) / self.width, (y0 - self.height / 2) / self.width
        r0 = np.sqrt(np.square(u0) + np.square(v0))
        theta = np.atan2(v0, u0)

        r1 = np.polyval(self.dist_poly, np.atan(r0))
        u1, v1 = r1 * np.cos(theta), r1 * np.sin(theta)
        x1, y1 = self.width * u1 + self.width / 2, self.width * v1 + self.height / 2
        return x1, y1

    def undistort(self, x0, y0):
        u0, v0 = (x0 - self.width / 2) / self.width, (y0 - self.height / 2) / self.width
        r0 = np.sqrt(np.square(u0) + np.square(v0))
        theta = np.atan2(v0, u0)

        r2 = np.tan(np.polyval(self.undist_poly, r0))
        u2, v2 = r2 * np.cos(theta), r2 * np.sin(theta)
        return u2, v2

    def coordinates_from_world_to_image(self, u, v):
        if len(u) <= 0 or len(v) <= 0:
            return np.array([])

        world_coords = np.stack([u, v, np.ones_like(u)])     # Shape: 3 x N
        H = self.camera_matrix[:, [0, 1, 3]]                 # Shape: 3 x 3
        image_coords = np.dot(H, world_coords)               # Shape: 3 x N

        # print('world = ', world_coords)
        # print('H = ', H)
        # print('image = ', image_coords / image_coords[[-1]])

        u, v, _ = image_coords / image_coords[[-1]]
        u, v = self.distort(u, v)
        return np.stack([u, v])

    def coordinates_from_image_to_world(self, x, y):
        if len(x) <= 0 or len(y) <= 0:
            return np.array([])

        x, y = self.undistort(x, y)

        img_coords = np.stack([x, y, np.ones_like(x)])      # Shape: 3 x N
        H = self.camera_matrix[:, [0, 1, 3]]                # Shape: 3 x 3
        world_coords = np.linalg.solve(H, img_coords)       # Shape: 3 x N

        x, y, _ = world_coords / world_coords[[-1]]
        return np.stack([x, y])

class COCOCategory(BaseModel):
    id: int
    name: str

class COCOModel(BaseModel):
    annotations: List[COCOAnnotation] = []
    categories: List[COCOCategory] = []
    images: List[COCOImage]