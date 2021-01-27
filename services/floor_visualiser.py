import numpy as np, cv2, colorsys, math
from services.texture_synthesizer import TextureSynthesizer
from PIL import Image

FLOOR_RGB_COLOR = (80, 50, 50)


class FloorVisualiser():
    def __init__(self, texture_synthesizer, floorRgbColor=(FLOOR_RGB_COLOR)):
        self.floorRgbValue = floorRgbColor
        self.texture_synthesizer = texture_synthesizer

    def replaceFloor(self, source, segmentedFloor, replacementTexture):
        """Given the semantic segmentation of the image's floor,
           replace the floor in an image with another texture provided.
        Args:
            source ([np.array]): [cv2 image in BGR format]
            segmentedFloor ([np.array]): [floor segmentation image of the source.
                                          floor can be segmented in any non-black color]
            replacementTexture ([np.array]): [replacement image used to tile the floor. Must be square in shape.]

        Raises:
            ValueError: [source and segmented floor images don't align in shape.]

        Returns:
            [np.array]: [source with floor replaced in rgba format.]
        """
        if (source.shape != segmentedFloor.shape):
            raise ValueError(
                "segmentedFloor does not have the same same shape as source")

        segmented_floor_grayscale = cv2.cvtColor(segmentedFloor,
                                                 cv2.COLOR_BGR2GRAY)
        source_image_with_rgba = cv2.cvtColor(source, cv2.COLOR_BGR2RGBA)
        replacement_texture_with_rgb = cv2.cvtColor(replacementTexture,
                                                    cv2.COLOR_BGR2RGB)
        _, segmented_floor_thresholded = cv2.threshold(segmented_floor_grayscale, 0, 255,
                          cv2.THRESH_BINARY)
        segmented_floor_filled = segmented_floor_thresholded
        # self.fillHoles(segmented_floor_thresholded)

        Image.fromarray(segmented_floor_thresholded).show()
        Image.fromarray(segmented_floor_filled).show()

        # self.boundSegmentedArea(segmented_floor_thresholded)  # not used

        mask = (segmented_floor_filled[:, :] != 0)
        synthesized_texture = self.texture_synthesizer.synthesizeTexture(
            replacement_texture_with_rgb, (source.shape[0], source.shape[1]))
        resized_synthesized_texture = cv2.cvtColor(
            cv2.resize(replacement_texture_with_rgb,
                       dsize=(segmented_floor_filled.shape[1],
                              segmented_floor_filled.shape[0])),
            cv2.COLOR_RGB2RGBA)

        source_image_with_rgba[:, :, :4][mask] = resized_synthesized_texture[
            mask]  # apply new flooring to source

        return source_image_with_rgba

    def fillHoles(self, grayscale_img, anchor_box=(3,3)):
        """
        Fill holes using morphological opening operation. 
        Args:
            grayscale_img ([np.array]): black and white img - preferably thresholded
            anchor_box (tuple[int], optional): sliding window convolution is done over. 
            Larger value means bigger holes are filled. Defaults to (3,3).
        Returns:
            [np.array]: [grayscale_img with holes filled.]
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, anchor_box)
        res = cv2.morphologyEx(grayscale_img, cv2.MORPH_OPEN, kernel)
        return res

    def boundSegmentedArea(self, segmentedFloorThresholded):
        """   
        @see: https://stackoverflow.com/questions/50051916/bounding-box-on-objects-based-on-color-python
        """
        segmented_floor_bgr = cv2.cvtColor(segmentedFloorThresholded,
                                           cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(segmentedFloorThresholded,
                                       cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(segmented_floor_bgr, contours, -1, (0, 255, 0), 2)
        for contour in contours:
            bbox = cv2.boundingRect(contour)
            cv2.rectangle(segmented_floor_bgr, bbox, (255, 0, 0), 1)

        Image.fromarray(segmented_floor_bgr).show()
        return segmented_floor_bgr