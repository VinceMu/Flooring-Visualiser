import random
import colorsys
import cv2
import numpy as np
import onnxruntime
from scipy.special import expit
from model.image_spec import ImageSpec
from model.nms_mode import NMS_Mode
from .onnx_inferer import OnnxInferer


class Yolo4ImageDetector(OnnxInferer):
    def __init__(self,
                 image_spec: ImageSpec,
                 inference_session: onnxruntime.InferenceSession,
                 anchors: np.ndarray,
                 classes: np.ndarray,
                 xyscale=(1.2, 1.1, 1.05),
                 strides=(8, 16, 32)):
        self.height_bound = image_spec.height
        self.width_bound = image_spec.width
        self.is_valid_input = image_spec.check_image
        self.run_time = inference_session
        self.anchors = anchors
        self.classes = classes
        self.xyscale = xyscale
        self.strides = strides
        super(Yolo4ImageDetector, self).__init__(inference_session, 1)

    def infer_once(self, image: np.array):
        h, w, _ = image.shape
        scale = min(self.width_bound / w, self.height_bound / h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))
        image_padded = np.full(shape=[self.height_bound, self.width_bound, 3],
                               fill_value=128.0)
        dw, dh = (self.width_bound - nw) // 2, (self.height_bound - nh) // 2
        image_padded[dh:nh + dh, dw:nw + dw, :] = image_resized
        
        image_padded = image_padded / 255.
        image_data = image_padded[np.newaxis,
                                  ...].astype(np.float32)  # add a dimension

        detections = self.infer_with_onnx([image_data])
        return self._process_post_inference(detections, image)

    def classify(self, image: np.array, bboxes, show_label=True):
        """
        bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
        """
        num_classes = self.classes.shape[
            0]  # classes is a one dimensional array
        image_h, image_w, _ = image.shape
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.)
                      for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        for i, bbox in enumerate(bboxes):
            coor = np.array(bbox[:4], dtype=np.int32)
            fontScale = 0.5
            score = bbox[4]
            class_ind = int(bbox[5])
            bbox_color = colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
            cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

            if show_label:
                bbox_mess = '%s: %.2f' % (self.classes[class_ind], score)
                t_size = cv2.getTextSize(bbox_mess,
                                         0,
                                         fontScale,
                                         thickness=bbox_thick // 2)[0]
                cv2.rectangle(image, c1,
                              (c1[0] + t_size[0], c1[1] - t_size[1] - 3),
                              bbox_color, -1)
                cv2.putText(image,
                            bbox_mess, (c1[0], c1[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 0, 0),
                            bbox_thick // 2,
                            lineType=cv2.LINE_AA)

        return image

    def _process_post_inference(self,
                                pred_bbox: list,
                                orig_image: np.array,
                                score_threshold=0.25):
        # go over raw detections with anchor boxes
        for i, pred in enumerate(pred_bbox):
            conv_shape = pred.shape
            output_size = conv_shape[1]
            conv_raw_dxdy = pred[:, :, :, :, 0:2]
            conv_raw_dwdh = pred[:, :, :, :, 2:4]
            xy_grid = np.meshgrid(np.arange(output_size),
                                  np.arange(output_size))
            xy_grid = np.expand_dims(np.stack(xy_grid, axis=-1), axis=2)

            xy_grid = np.tile(np.expand_dims(xy_grid, axis=0), [1, 1, 1, 3, 1])
            xy_grid = xy_grid.astype(np.float)

            pred_xy = ((expit(conv_raw_dxdy) * self.xyscale[i]) - 0.5 *
                       (self.xyscale[i] - 1) + xy_grid) * self.strides[i]
            pred_wh = (np.exp(conv_raw_dwdh) * self.anchors[i])
            pred[:, :, :, :, 0:4] = np.concatenate([pred_xy, pred_wh], axis=-1)

        pred_bbox = [np.reshape(x, (-1, np.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = np.concatenate(pred_bbox, axis=0)
        # remove boundary boxs with a low detection probability
        valid_scale = [0, np.inf]
        pred_bbox = np.array(pred_bbox)

        pred_xywh = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate([
            pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
            pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5
        ],
                                   axis=-1)
        # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        org_h, org_w = orig_image.shape[:2]
        resize_ratio = min(self.width_bound / org_w, self.height_bound / org_h)

        dw = (self.width_bound - resize_ratio * org_w) / 2
        dh = (self.height_bound - resize_ratio * org_h) / 2

        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # # (3) clip some boxes that are out of range
        pred_coor = np.concatenate([
            np.maximum(pred_coor[:, :2], [0, 0]),
            np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])
        ],
                                   axis=-1)
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]),
                                     (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        # # (4) discard some invalid boxes
        bboxes_scale = np.sqrt(
            np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale),
                                    (bboxes_scale < valid_scale[1]))

        # # (5) discard some boxes with low scores
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > score_threshold
        mask = np.logical_and(scale_mask, score_mask)
        coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

        res = np.concatenate(
            [coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

        # # result shape is (x, 6) where x is number of detections and 6 is composed of
        # # (xmin, ymin, xmax, ymax, score, class)
        post_nms = np.array(self._nms(res, 0.213))
        print(post_nms.shape)
        print(post_nms[post_nms[:,4].argsort()])
        return post_nms

    def _bboxes_iou(self, boxes1, boxes2):
        '''calculate the Intersection Over Union value'''
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] -
                                                           boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] -
                                                           boxes2[..., 1])

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        ious = np.maximum(1.0 * inter_area / union_area,
                          np.finfo(np.float32).eps)

        return ious

    def _nms(self, bboxes, iou_threshold, sigma=0.3, method=NMS_Mode.STANDARD):
        """
        :param bboxes: (xmin, ymin, xmax, ymax, score, class)

        Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
            https://github.com/bharatsingh430/soft-nms
        """
        classes_in_img = list(set(bboxes[:, 5]))
        best_bboxes = []

        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]

            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate(
                    [cls_bboxes[:max_ind], cls_bboxes[max_ind + 1:]])
                iou = self._bboxes_iou(best_bbox[np.newaxis, :4],
                                       cls_bboxes[:, :4])
                weight = np.ones((len(iou), ), dtype=np.float32)

                if method == NMS_Mode.STANDARD:
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0

                elif method == NMS_Mode.SOFT:
                    weight = np.exp(-(1.0 * iou**2 / sigma))

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]

        return best_bboxes


class MisalignedInputError(Exception):
    pass


def get_anchors(path: str):
    with open(path, 'r') as f:
        line = f.readline()
    anchors = np.array(line.split(','), dtype=np.float32)
    return anchors.reshape((3, 3, 2))


def get_classes(path: str):
    with open(path, 'r') as f:
        classes = f.readlines()
    return np.array(classes, dtype=object)
