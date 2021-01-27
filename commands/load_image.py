from .command_base import CommandBase
import cv2


class LoadImageCommand(CommandBase):
    def __init__(self):
        super(LoadImageCommand, self).__init__(
            "load_image",
            prereq=None,
            action="store",
            help="Load Image accepts the path of the image to be scored")

    def _run(self, arg, previous_output):
        return (cv2.cvtColor(cv2.imread(arg), cv2.COLOR_BGR2RGB), arg)
