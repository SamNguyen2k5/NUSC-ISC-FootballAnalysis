import litserve as ls
from .inference.segmentation import SegmentationInference

server = ls.LitServer(SegmentationInference(max_batch_size=1), accelerator="auto")
server.run(port=8001)