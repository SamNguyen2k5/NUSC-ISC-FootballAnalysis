import litserve as ls
from .inference.segmentation import SegmentationInferenceAPI
from .inference.spiideo import SpiideoVisualiserAPI

server = ls.LitServer(
    [
        SegmentationInferenceAPI(max_batch_size=1, api_path='/segmentation/predict'),
        SpiideoVisualiserAPI(max_batch_size=1, api_path='/spiideo/visualise'),
    ],
    accelerator="auto"
)

server.run(port=8001)