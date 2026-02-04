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

class COCOImage(BaseModel):
    id: int
    file_name: str
    width: int
    height: int
    camera_matrix: NDArray[Shape['3, 4'], float]            # type: ignore
    dist_poly: NDArray[Shape['9'], float]                   # type: ignore
    undist_poly: NDArray[Shape['9'], float]                 # type: ignore

class COCOCategory(BaseModel):
    id: int
    name: str

class COCOModel(BaseModel):
    annotations: List[COCOAnnotation]
    categories: List[COCOCategory]
    images: List[COCOImage]