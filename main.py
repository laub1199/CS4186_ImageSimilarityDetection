from __future__ import division
import cv2
import numpy as np
import time
from PIL import Image
from collections import Counter
from matplotlib import pyplot as plt

def SIFT_BF(query, compare):
    query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    compare = cv2.cvtColor(compare, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(query, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(compare, None)

    # L1 0.0018
    # L2 0.0030 good/matches 0.8
    # cos sim 0.004477
    bf = cv2.BFMatcher()

    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    # matches = sorted(matches, key=lambda x:x.distance)

    good = []
    m_dis = 0
    n_dis = 0
    dot = 0
    for m, n in matches:
        if m.distance < n.distance * 0.8:
            good.append([m])
    return (len(good)/len(matches))
            # if m.distance * n.distance < 0.8 :
            #         dot += m.distance * n.distance
            #         m_dis += m.distance**2
            #         n_dis += n.distance**2
            # return dot / (np.sqrt(m_dis) * np.sqrt(n_dis))

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

    return cv2.compareHist(hist_query, hist_compare, 0)

    # for compare_method in range(4):
    #     query_query = cv2.compareHist(hist_query, hist_query, compare_method)
    #     query_half = cv2.compareHist(hist_query, hist_half_down, compare_method)
    #     query_compare = cv2.compareHist(hist_query, hist_compare, compare_method)
    #
    #     print('Method:', compare_method, 'Perfect, Base-Half, Image-Compare :', query_query, '/', query_half, '/', query_compare)

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

def ORB_BF(queryPath, comparePath):
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

    img1 = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(compare, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()

    keypoints_1, descriptors_1 = orb.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    if (descriptors_1 is not None and descriptors_2 is not None):
        matches = bf.match(descriptors_1, descriptors_2)

        similar_regions = [i for i in matches if i.distance < 50]
        if len(matches) == 0:
            return 0
        return len(similar_regions) / len(matches)
    return 0

def ORB_BFKNN(queryPath, comparePath):
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

    img1 = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(compare, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()

    keypoints_1, descriptors_1 = orb.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=False)

    if (descriptors_1 is not None and descriptors_2 is not None):
        matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)
        good = []
        if matches is not None and len(matches[0]) > 1:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append([m])
            return len(good)/len(matches)
        return 0
    return 0
        # matchingImage = cv2.drawMatchesKnn(query, keypoints_1, compare, keypoints_2, good, compare, flags=2)

        # cv2.imshow("Image", matchingImage)
        # cv2.waitKey(0)

def Euclidean(queryPath, comparePath):
    # smaller better
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

    query_arr = np.asarray(query)
    query_arr_flat = query_arr.flatten()
    query_RH = Counter(query_arr_flat)
    query_H = []
    for i in range(256):
        if i in query_RH.keys():
            query_H.append(query_RH[i])
        else:
            query_H.append(0)

    compare_arr = np.asarray(compare)
    compare_arr_flat = compare_arr.flatten()
    compare_RH = Counter(compare_arr_flat)
    compare_H = []
    for i in range(256):
        if i in compare_RH.keys():
            compare_H.append(compare_RH[i])
        else:
            compare_H.append(0)

    return L2Norm(query_H, compare_H)

def L2Norm(H1,H2):
    distance =0
    for i in range(len(H1)):
        distance += (np.square(np.array(H1[i]-H2[i]), dtype='int64'))
    return np.sqrt(distance)

def COS(queryPath, comparePath):
    # bigger better
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

    query_arr = np.asarray(query)
    query_arr_flat = query_arr.flatten()
    query_RH = Counter(query_arr_flat)
    query_H = []
    for i in range(256):
        if i in query_RH.keys():
            query_H.append(query_RH[i])
        else:
            query_H.append(0)

    compare_arr = np.asarray(compare)
    compare_arr_flat = compare_arr.flatten()
    compare_RH = Counter(compare_arr_flat)
    compare_H = []
    for i in range(256):
        if i in compare_RH.keys():
            compare_H.append(compare_RH[i])
        else:
            compare_H.append(0)

    return dot(query_H, compare_H) / (norm(query_H) * norm(compare_H))

def dot(H1,H2):
    distance =0
    for i in range(len(H1)):
        distance += H1[i] * H2[i]
    return distance

def norm(H):
    distance =0
    for i in range(len(H)):
        distance += (np.square(np.array(H[i]), dtype='int64'))
    return np.sqrt(distance)

def SIFT_FLANN(queryPath, comparePath):
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


    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(query, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(compare, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    return len(draw_params)

    img3 = cv2.drawMatchesKnn(query,keypoints_1,compare,keypoints_2,matches,None,**draw_params)

    plt.imshow(img3, 'gray'), plt.show()

def MSE(query, compare):
    query = cv2.resize(query, (500, 500), interpolation=cv2.INTER_AREA)
    compare = cv2.resize(compare, (500, 500), interpolation=cv2.INTER_AREA)
    error_rate = np.sum((compare.astype('float') - query.astype('float')) ** 2)
    error_rate = error_rate / (float(compare.shape[0] * compare.shape[1]))
    return error_rate

########################################################################################################################
#                                                                                                                      #
#                                                                                                                      #
#                                              All methods end here                                                    #
#                                              All methods end here                                                    #
#                                              All methods end here                                                    #
#                                              All methods end here                                                    #
#                                              All methods end here                                                    #
#                                              All methods end here                                                    #
#                                                                                                                      #
#                                                                                                                      #
########################################################################################################################

def temp():
    text_file = open("6-SIFT_LENGTH_10_example_rankList.txt", "w")
    data_file = open("6-SIFT_LENGTH_10_example_data.txt", "w")
    # queryPath = "C:/Users/laub1/Desktop/4186/Queries/"
    queryPath = "C:/Users/laub1/Desktop/4186/examples/example_query/"
    comparePath = "C:/Users/laub1/Desktop/4186/Images/"
    list = []


    # for i in range(253, 1665):
    #     compareNum = "0000" + str(i)
    #     compareNum = compareNum[-4:]
    #     similarity = SIFT_FLANN(queryPath+'01', comparePath+compareNum)
    #     print(compareNum + ': ' + str(similarity))
    #     temp = [i, similarity]
    #     list.append(temp)
    # list = sorted(list, key=lambda x: x[1], reverse=True)
    # print(list)

    # YOLO(comparePath + '2595', 0.5, 0.3)
    #
    # # Hist(queryPath + '01', comparePath + '0017')

    # SIFT_BF(queryPath+"01", comparePath+"0017")

    # ORB_BFKNN(queryPath+"02", comparePath+"3451")
    for q in range(1, 11):
        queryNum = "00" + str(q)
        queryNum = queryNum[-2:]
        for i in range(252, 5001):
            compareNum = "0000" + str(i)
            compareNum = compareNum[-4:]
            similarity = SIFT_BF(queryPath+queryNum, comparePath+compareNum)
            print(queryNum + '-' + compareNum + ': ' + str(similarity))
            temp = [i, similarity]
            list.append(temp)
            data_file.write(str(q) + ' ' + compareNum + ' ' + str(similarity) + '\n')

        # reverse true = descending
        list = sorted(list, key=lambda x: x[1], reverse=False)

        text_file.write('Q' + str(q) + ': ')
        for i in range(0, len(list)):
            text_file.write(str(list[i][0]))
            text_file.write(' ')
        text_file.write('\n')
        list = []
    text_file.close()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def reverse():
    path = '18-MSE_10_example_500_500_rankList'
    file = open("./" + path + ".txt").read().splitlines()
    out = open(path+"_reverse.txt", "w")
    for line in file:
        strlist = line.split()
        temp = []
        for str in strlist:
            if 'Q' not in str:
                temp.append(str)
            else:
                out.write(str)
        temp.reverse()
        for str in temp:
            out.write(' ' + str)
        out.write('\n')

def ave_pre(array):
    ap = 0
    for idx, m in enumerate(array):
        ap_m = (idx + 1) / (m + 1)
        ap = ap + ap_m
    ap = ap / len(array)
    return ap

def check_precision(rank_result, till_q = 11):
    # compute mean average precision
    rank_line = open('rankList.txt').read().splitlines()
    ap_sum = 0
    for idx, line in enumerate(rank_result):
        if idx > till_q:
            break
        line_str = line.split()
        query_num = int(line_str[0][1]) - 1
        result_num = [int(x) for x in line_str[1:]]
        rank_str = rank_line[idx].split()
        rank_gt = [int(x) for x in rank_str[1:]]
        find_idx = []
        for num in rank_gt:
            ind = np.where(np.array(result_num) == num)
            find_idx.extend(ind)
        find_idx = np.array(find_idx).reshape(len(find_idx), )
        find_idx = np.sort(find_idx)
        ap = ave_pre(find_idx)
        print("Average Precision of Q%d: %.4f" % (idx + 1, ap))
        ap_sum = ap_sum + ap
    print("Mean Average Precision: %f" % (ap_sum / len(rank_result)))

def getRankList(method, index, q_from=1, q_to=11, c_from=1, c_to=5001, extraText='', isExample=True):
    file_extends = "_10_example_" if isExample else "_" + str(q_to) + "_"
    text_file = open(str(index) + "-" + method + file_extends + extraText + "rankList.txt", "w")
    data_file = open(str(index) + "-" + method + file_extends + extraText + "data.txt", "w")
    queryPath = "C:/Users/laub1/Desktop/4186/examples/example_query/" if isExample else "C:/Users/laub1/Desktop/4186/Queries/"
    comparePath = "C:/Users/laub1/Desktop/4186/Images/"
    list = []

    for q in range(q_from, q_to):
        queryNum = "00" + str(q)
        queryNum = queryNum[-2:]
        for i in range(c_from, c_to):
            compareNum = "0000" + str(i)
            compareNum = compareNum[-4:]
            similarity = None

            query = cv2.imread(queryPath + queryNum + ".jpg")
            compare = cv2.imread(comparePath + compareNum + ".jpg")

            f1 = open(queryPath + queryNum + ".txt").read().split()

            x1 = int(f1[0])
            y1 = int(f1[1])
            w1 = int(f1[2])
            h1 = int(f1[3])
            query = query[y1:y1 + h1, x1:x1 + w1]

            if int(str(comparePath + compareNum).split('Images/')[1]) <= 2000:
                f2 = open(comparePath + compareNum + ".txt").read().split()

                x2 = int(f2[0])
                y2 = int(f2[1])
                w2 = int(f2[2])
                h2 = int(f2[3])
                compare = compare[y2:y2 + h2, x2:x2 + w2]

            if method == "SIFT_BF":
                similarity = SIFT_BF(query, compare)
            if method == 'MSE':
                similarity = MSE(query, compare)


            # orb = ORB_BF(queryPath+queryNum, comparePath+compareNum)
            # if orb == 0:
            #     similarity = similarity * ORB_BF(queryPath+queryNum, comparePath+compareNum)
            print(queryNum + '-' + compareNum + ': ' + str(similarity))
            temp = [i, similarity]
            list.append(temp)
            data_file.write(str(q) + ' ' + compareNum + ' ' + str(similarity) + '\n')


        # reverse true = descending
        list = sorted(list, key=lambda x: x[1], reverse=False)

        text_file.write('Q' + str(q) + ': ')
        for i in range(0, len(list)):
            text_file.write(str(list[i][0]))
            text_file.write(' ')

        text_file.write('\n')
        list = []

    text_file.close()
    rankList = open(str(index) + "-" + method + file_extends + "rankList.txt").read().splitlines()
    return rankList


if __name__ == '__main__':
    # reverse()
    rankList = getRankList('MSE', 18, 1, 3, 1, 5001,'500_500_')
    # check_precision(rankList, 1)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
