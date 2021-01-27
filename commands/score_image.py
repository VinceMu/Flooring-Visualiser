from services.yolo4_image_detector import Yolo4ImageDetector
from .command_base import CommandBase
from .load_image import LoadImageCommand


class ScoreImageCommand(CommandBase):
    def __init__(self, loadImage: LoadImageCommand,
                 scorer: Yolo4ImageDetector):
        self.scorer = scorer
        super(ScoreImageCommand, self).__init__(
            "--score_image",
            prereq=loadImage,
            action='store_true',
            help="ScoreImage accepts path of image to be scored")

    def _run(self, arg, previous_output):
        previous_output, _ = previous_output
        return (previous_output, self.scorer.infer_once(previous_output))
