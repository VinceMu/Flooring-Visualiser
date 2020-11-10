from services.command_handler import CommandHandler
from model import ImageSpec
from services.yolo4_image_detector import Yolo4ImageDetector, get_anchors, get_classes
from commands.score_image import ScoreImageCommand
from commands.display_classifications import DisplayClassificationsCommand
from commands.load_image import LoadImageCommand
from onnxruntime import InferenceSession

yolo4_inferer = Yolo4ImageDetector(
    image_spec=ImageSpec(416, 416),
    inference_session=InferenceSession("./data/onnx/yolov4/yolov4.onnx"),
    anchors=get_anchors("./data/onnx/yolov4/anchors.txt"),
    classes=get_classes("./data/onnx/yolov4/coco.names"))

load_image_cmd = LoadImageCommand()
score_image_cmd = ScoreImageCommand(load_image_cmd, yolo4_inferer)
classify_image_cmd = DisplayClassificationsCommand(score_image_cmd,
                                                   yolo4_inferer.classify)

if __name__ == "__main__":
    commandList = [load_image_cmd, score_image_cmd, classify_image_cmd]
    parser = CommandHandler()
    for cmd in commandList:
        parser.addCommand(cmd)
    result = parser.run_commands()
"""
test usage: python app.py --classify_image ./data/dog.jpg
"""
