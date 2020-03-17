import numpy as np
import torch

'''
    Funciton: nms function to wipe out redundant rect
    Author: cema
    Date: 2020.03.17, Tuesday
    @boxes:          candidate boxes
    @scores:         scores for the candidate boxes
    @iou_threshold:  threshold for the iou
'''
def torch_nms(boxes, scores, iou_threshold):
    boxes_list = boxes.detach().numpy()

    x1 = boxes_list[:, 0]
    y1 = boxes_list[:, 1]
    x2 = boxes_list[:, 2]
    y2 = boxes_list[:, 3]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    keep = []

    score_list = scores.detach().numpy().tolist()
    reverse_score_list = [-score for score in score_list]
    reverse_scores = torch.Tensor(reverse_score_list)

    index_tensor = reverse_scores.argsort()
    index = index_tensor.detach().numpy()

    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w * h

        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= iou_threshold)[0]

        index = index[idx + 1]  # because index start from 1
    return keep

def py_cpu_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []

    index = scores.argsort()[::-1]

    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w * h

        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= thresh)[0]

        index = index[idx + 1]  # because index start from 1
    return keep



'''
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def plot_bbox(dets, c = 'k'):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        plt.plot([x1, x2], [y1, y1], c)
        plt.plot([x1, x1], [y1, y2], c)
        plt.plot([x1, x2], [y2, y2], c)
        plt.plot([x2, x2], [y1, y2], c)
        plt.title("after nms")

    boxes = np.array([[100, 100, 210, 210, 0.72],
                      [250, 250, 420, 420, 0.8],
                      [220, 220, 320, 330, 0.92],
                      [100, 100, 210, 210, 0.72],
                      [230, 240, 325, 330, 0.81],
                      [220, 230, 315, 340, 0.9]])
    #plot_bbox(dets=boxes)
    keep = py_cpu_nms(boxes, thresh=0.7)
    plot_bbox(dets=boxes[keep])
    plt.show()
'''