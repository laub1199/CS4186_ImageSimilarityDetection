import cv2
import numpy as np
import time
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def SIFT_BF(queryPath, comparePath):
    query = cv2.imread(queryPath + ".jpg")
    compare = cv2.imread(comparePath + ".jpg")

    f1 = open(queryPath + ".txt").read().split()

    x1 = int(f1[0])
    y1 = int(f1[1])
    w1 = int(f1[2])
    h1 = int(f1[3])
    query = query[y1:y1 + h1, x1:x1 + w1]

    if int(str(comparePath).split('Images/')[1]) <= 2000:
        f2 = open(comparePath + ".txt").read().split()

        x2 = int(f2[0])
        y2 = int(f2[1])
        w2 = int(f2[2])
        h2 = int(f2[3])
        compare = compare[y2:y2 + h2, x2:x2 + w2]

    query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    compare = cv2.cvtColor(compare, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(query, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(compare, None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)

    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append([m])

    print(len(good)/len(matches))

    matchingImage = cv2.drawMatchesKnn(query, keypoints_1, compare, keypoints_2, good, compare, flags=2)

    cv2.imshow("Image", matchingImage)
    cv2.waitKey(0)

def Hist(queryPath, comparePath):
    query = cv2.imread(queryPath + '.jpg')
    compare = cv2.imread(comparePath + '.jpg')

    if query is None or compare is None:
        print('Couldn\'t read image')
        return

    f1 = open(queryPath + ".txt").read().split()

    x1 = int(f1[0])
    y1 = int(f1[1])
    w1 = int(f1[2])
    h1 = int(f1[3])
    query = query[y1:y1 + h1, x1:x1 + w1]

    if int(str(comparePath).split('Images/')[1]) <= 2000:
        f2 = open(comparePath + ".txt").read().split()

        x2 = int(f2[0])
        y2 = int(f2[1])
        w2 = int(f2[2])
        h2 = int(f2[3])
        compare = compare[y2:y2 + h2, x2:x2 + w2]


    # hsv_query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    # hsv_compare = cv2.cvtColor(compare, cv2.COLOR_BGR2GRAY)

    half_down = query[query.shape[0]//2:,:]

    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]

    channels = [0, 1]

    # hue varies from 0 to 179, saturation from 0 to 255
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges  # concat lists

    hist_query = cv2.calcHist([query], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_query, hist_query, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_half_down = cv2.calcHist([half_down], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_half_down, hist_half_down, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_compare = cv2.calcHist([compare], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_compare, hist_compare, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    for compare_method in range(4):
        query_query = cv2.compareHist(hist_query, hist_query, compare_method)
        query_half = cv2.compareHist(hist_query, hist_half_down, compare_method)
        query_compare = cv2.compareHist(hist_query, hist_compare, compare_method)

        print('Method:', compare_method, 'Perfect, Base-Half, Image-Compare :', query_query, '/', query_half, '/', query_compare)

def YOLO(imagePath, confidence, threshold):
    # confidence default 0.5, threshold default 0.3
    LABELS = open('./yolo-coco/coco.names').read().strip().split('\n')

    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet('./yolo-coco/yolov3.cfg', './yolo-coco/yolov3.weights')

    img = cv2.imread(imagePath + '.jpg')
    (H, W) = img.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            _confidence = scores[classID]
            if _confidence > confidence:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # for box in boxes:
    #     (x, y) = (box[0], box[1])
    #     (w, h) = (box[2], box[3])
    #     color = [int(c) for c in COLORS[classIDs[0]]]
    #     cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    # print(boxes)
    # print(confidences)
    # print(confidence)
    # print(threshold)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)

    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Image", img)
    cv2.waitKey(0)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    queryPath = "C:/Users/laub1/Desktop/4186/Queries/"
    comparePath = "C:/Users/laub1/Desktop/4186/Images/"

    # YOLO(comparePath + '2595', 0.5, 0.3)
    #
    # # Hist(queryPath + '01', comparePath + '0017')

    SIFT_BF(queryPath+"01", comparePath+"0017")

    # for i in range(1, 2000):
    #     image = "0000" + str(i)
    #     image = image[-4:]
    #     SIFT_BF("01", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
