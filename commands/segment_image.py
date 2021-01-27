from services.semantic_segmenter import SemanticSegmenter
from .command_base import CommandBase
from .load_image import LoadImageCommand


class SegmentImageCommand(CommandBase):
    def __init__(self, loadImage: LoadImageCommand,
                 segmenter: SemanticSegmenter):
        self.segmenter = segmenter
        super(SegmentImageCommand,
              self).__init__("--segment_image",
                             prereq=loadImage,
                             action='store_true',
                             help="indicates an image should be segmented")

    def _run(self, arg, previous_output):
        image_array, image_path = previous_output
        res = self.segmenter.segmentImage(image_array)
        self.segmenter.encodeImageAndSave(
            res, image=None,
            imageName=image_path)  # no side by side comparison
        return (previous_output, res)
