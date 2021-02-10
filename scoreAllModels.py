import os, re, cv2
from pathlib import Path
from PIL import Image
from services.semantic_segmenter import SemanticSegmenter
from services.floor_visualiser import FloorVisualiser
from services.texture_synthesizer import TextureSynthesizer
from model import EncoderParams, DecoderParams
from utils import provide_model_params

MODEL_FOLDER = "./data/semantic_segmentation"
RESULT_FOLDER = "./data/segmentation_results"
IMAGES = [
    "./data/segmentation_test_set/living_space.jpg",
    "./data/segmentation_test_set/room.jpg",
    "./data/segmentation_test_set/wooden_living_room.jpg"
]

images_array = [(i.split("/")[-1], cv2.imread(i)) for i in IMAGES]
replacement_texture = sample_texture = cv2.imread(
    "./data/texture_synthesis_test_set/wooden_floor_board.jpg")

model_folders = [f.path for f in os.scandir(MODEL_FOLDER) if f.is_dir()]

texture_synthesizer = TextureSynthesizer(1.2)
floor_visualiser = FloorVisualiser(texture_synthesizer)

for model_dir in model_folders:
    encoder_path = [os.path.join(model_dir, i) for i in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir,i)) and \
         re.match(r'encoder\w+.pth', i) is not None][0]
    decoder_path = [os.path.join(model_dir, i) for i in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir,i)) and \
         re.match(r'decoder\w+.pth', i) is not None][0]

    model_fullname = model_dir.replace("\\", "/").split("/")[-1]
    _, encoder_name, decoder_name = model_fullname.split("-")
    print(f"scoring for {encoder_name} {decoder_name}")
    encoder_props, decoder_props = provide_model_params(model_dir)
    encoder_params = EncoderParams(arch=encoder_name,
                                   weights=encoder_path,
                                   **encoder_props)

    decoder_params = DecoderParams(arch=decoder_name,
                                   weights=decoder_path,
                                   **decoder_props)
    model = SemanticSegmenter(resultPath=RESULT_FOLDER,
                              encoderParams=encoder_params,
                              decoderParams=decoder_params)
    for img_obj in images_array:
        print(f"using {img_obj[0]} as source image")
        floor_segmentation = model.segmentFloor(img_obj[1])
        replaced_floor = floor_visualiser.replaceFloor(img_obj[1],
                                                       floor_segmentation,
                                                       replacement_texture)
        replaced_floor_dir = f"{RESULT_FOLDER}/{model_fullname}"

        Path(replaced_floor_dir).mkdir(parents=True, exist_ok=True)
        Image.fromarray(replaced_floor).save(
            f"{replaced_floor_dir}/{img_obj[0]}.png")
