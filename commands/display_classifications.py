from PIL import Image
from .command_base import CommandBase
from .score_image import ScoreImageCommand

STRIDES = [8, 16, 32]


class DisplayClassificationsCommand(CommandBase):
    def __init__(self, prereq: ScoreImageCommand, classify_func: callable):
        self.classify = classify_func
        super(DisplayClassificationsCommand, self).__init__(
            "--display",
            prereq=prereq,
            action="store_true",
            help="ClassifyImage accepts path of the classification dictionary")

    def _run(self, arg, previous_output):
        classifications = self.classify(previous_output[0], previous_output[1])
        image_from_classifications = Image.fromarray(classifications)
        image_from_classifications.show()
