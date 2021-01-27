from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode
from pathlib import Path
from PIL import Image
from model import EncoderArch, DecoderArch
import csv, torch, torchvision, numpy as np, scipy.io
"""
TODO:
 - consider refactoring colorPath out and passing in color dictionary as an __init__ param
 - batching
"""


class SemanticSegmenter():
    def __init__(self,
                 encoderArch: EncoderArch,
                 decoderArch: DecoderArch,
                 encoderWeightsPath: str,
                 decoderWeightsPath: str,
                 classesPath:
                 str = "./data/semantic_segmentation/object150_info.csv",
                 colorPath: str = "./data/semantic_segmentation/color150.mat",
                 resultPath: str = "./data/segmentation_results"):
        encoder = ModelBuilder.build_encoder(arch=encoderArch,
                                             fc_dim=2048,
                                             weights=encoderWeightsPath)
        decoder = ModelBuilder.build_decoder(arch=decoderArch,
                                             fc_dim=2048,
                                             num_class=150,
                                             weights=decoderWeightsPath,
                                             use_softmax=True)
        encoderName = encoderWeightsPath.split("/")[
            -2]  # encompassing folder is the name
        self.resultPath = f'{resultPath}/{encoderName}/'
        Path(resultPath).mkdir(parents=True, exist_ok=True)

        crit = torch.nn.NLLLoss(ignore_index=-1)
        self.segmentation_module = SegmentationModule(encoder, decoder, crit)
        self.segmentation_module.eval()
        self.pil_to_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # These are RGB mean+std values
                std=[0.229, 0.224, 0.225])  # across a large photo dataset.
        ])
        self.colors = scipy.io.loadmat(colorPath)['colors']
        self.classes = self._load_classes(classesPath)
        self.floor_index = [
            k for k, v in self.classes.items() if v == 'floor'
        ][0] - 1  # should only be one floor class

    def _load_classes(self, classesPath: str):
        names = {}
        with open(classesPath) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                names[int(row[0])] = row[5].split(";")[0]
        return names

    def segmentImage(self, image):
        img_data = self.pil_to_tensor(image)
        singleton_batch = {'img_data': img_data[None]}
        output_size = img_data.shape[1:]
        # Run the segmentation at the highest resolution.
        with torch.no_grad():
            scores = self.segmentation_module(singleton_batch,
                                              segSize=output_size)
        # Get the predicted scores for each pixel
        _, pred = torch.max(scores, dim=1)
        pred = pred.cpu()[0].numpy()
        return pred

    def _encodePrediction(self, pred: np.array, index: np.array):
        # filter prediction class if requested
        if index is not None:
            pred = pred.copy()
            pred[pred != index] = -1
            print(f'{self.classes[index+1]}:')

        # colorize prediction
        pred_color = colorEncode(pred, self.colors).astype(np.uint8)
        print(pred_color)
        return pred_color

    def encodeImageAndSave(self,
                           pred: np.array,
                           image: np.array = None,
                           imageName: str = "test.jpeg",
                           index: np.array = None):
        # colorize prediction
        pred_color = self._encodePrediction(pred, index)

        # optionally aggregate images and save
        im_vis = np.concatenate(
            (image, pred_color), axis=1) if image is not None else pred_color

        segmented_image = Image.fromarray(im_vis)
        # max quality and min subsampling to eliminate noise - we want sharpest edges
        segmented_image.save(self.resultPath + imageName.split("/")[-1],
                             quality=100,
                             subsampling=0)

    def encodeImageAndShow(self,
                           pred: np.array,
                           image: np.array = None,
                           index: np.array = None):
        # colorize prediction
        pred_color = self._encodePrediction(pred, index)

        # optionally aggregate images and save
        im_vis = np.concatenate(
            (image, pred_color), axis=1) if image is not None else pred_color
        (Image.fromarray(im_vis)).show()


    def encodeFloorAndSave(self,
                           pred: np.array,
                           image: np.array = None,
                           imageName: str = "test.jpeg"):
        self.encodeImageAndSave(pred, image, imageName, self.floor_index)

    def visualiseTopTen(self, image: np.array, pred: np.array):
        # Top classes in answer
        predicted_classes = np.bincount(pred.flatten()).argsort()[::-1]
        for c in predicted_classes[:15]:
            self.encodeImageAndShow(pred, image, c)
