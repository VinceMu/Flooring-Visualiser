from commands import LoadImageCommand, SegmentImageCommand, ScoreImageCommand, DisplayClassificationsCommand, SegmentFloorCommand
from services.command_handler import CommandHandler
from model import ImageSpec, EncoderArch, DecoderArch
from services.yolo4_image_detector import Yolo4ImageDetector, get_anchors, get_classes
from services.semantic_segmenter import SemanticSegmenter
from onnxruntime import InferenceSession

yolo4_inferer = Yolo4ImageDetector(
    image_spec=ImageSpec(416, 416),
    inference_session=InferenceSession("./data/onnx/yolov4/yolov4.onnx"),
    anchors=get_anchors("./data/onnx/yolov4/anchors.txt"),
    classes=get_classes("./data/onnx/yolov4/coco.names"))

semantic_segmenter = SemanticSegmenter(
    encoderArch=EncoderArch.resnet101dilated,
    decoderArch=DecoderArch.ppm_deepsup,
    encoderWeightsPath=
    "./data/semantic_segmentation/ade20k-resnet101dilated-ppm_deepsup/encoder_epoch_25.pth",
    decoderWeightsPath=
    "./data/semantic_segmentation/ade20k-resnet101dilated-ppm_deepsup/decoder_epoch_25.pth",
)

load_image_cmd = LoadImageCommand()
score_image_cmd = ScoreImageCommand(load_image_cmd, yolo4_inferer)
classify_image_cmd = DisplayClassificationsCommand(score_image_cmd,
                                                   yolo4_inferer.classify)
segment_image_cmd = SegmentImageCommand(load_image_cmd, semantic_segmenter)
segment_floor_cmd = SegmentFloorCommand(load_image_cmd, semantic_segmenter)

if __name__ == "__main__":
    commandList = [
        load_image_cmd, score_image_cmd, classify_image_cmd, segment_image_cmd, segment_floor_cmd
    ]
    parser = CommandHandler()
    for cmd in commandList:
        parser.addCommand(cmd)
    result = parser.run_commands()
"""
test usage: python app.py --score_image ./data/dog.jpg
"""
