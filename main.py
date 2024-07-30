import os
import cv2
import numpy as np
import random


def get_detections(net, blob):
    net.setInput(blob)
    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    return boxes, masks


# Define Paths
cfg_path = r"model\mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
weights_path = r"model\frozen_inference_graph.pb"
class_names_path = r"model\class.names"

img_path = r"cat_and_dog.png"

# Load image
img = cv2.imread(img_path)
H, W, C = img.shape

# Load Model
net = cv2.dnn.readNetFromTensorflow(weights_path, cfg_path)

# Convert Image
blob = cv2.dnn.blobFromImage(img)

# Get mask
boxes, masks = get_detections(net, blob)

# Draw Masks
colors = [
    (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for j in range(90)
]
empty_image = np.zeros((H, W, C))

# print(len(masks))
detection_thresh = 0.5

for j in range(len(masks)):
    bbox = boxes[0, 0, j]

    class_id = bbox[1]
    score = bbox[2]

    if score > detection_thresh:
        # print(bbox)
        x1, y1, x2, y2 = (
            int(bbox[3] * W),
            int(bbox[4] * H),
            int(bbox[5] * W),
            int(bbox[6] * H),
        )

        # print(int(class_id))
        # img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        mask = masks[j]

        mask = mask[int(class_id)]

        # print(mask.shape)
        # print(H, W)

        mask = cv2.resize(mask, (x2 - x1, y2 - y1))

        _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
        # print(mask.shape)
        for c in range(3):
            empty_image[y1:y2, x1:x2, c] = mask * colors[int(class_id)][c]

overlay = ((0.6 * empty_image) + (0.4 * img)).astype("uint8")

# Visualisation
# cv2.imshow("mask", empty_image)
# cv2.imshow("img", img)
cv2.imshow("semantic_segmented_img", overlay)
cv2.imwrite("semantic_segmented_img.png", overlay)

cv2.waitKey(0)
cv2.destroyAllWindows()
