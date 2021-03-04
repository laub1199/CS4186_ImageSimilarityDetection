import cv2
import numpy as np
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def SIFT_BF(num1, num2):
    img1 = cv2.imread("C:/Users/laub1/Desktop/4186/Queries/" + num1 + ".jpg")
    img2 = cv2.imread("C:/Users/laub1/Desktop/4186/Images/" + num2 + ".jpg")

    if (int(num2) <= 2000):
        f1 = open("C:/Users/laub1/Desktop/4186/Queries/" + num1 + ".txt").read().split()
        f2 = open("C:/Users/laub1/Desktop/4186/Images/" + num2 + ".txt").read().split()
        x1 = int(f1[0])+20
        y1 = int(f1[1])+20
        w1 = int(f1[2])-40
        h1 = int(f1[3])-40
        img1 = img1[y1:y1 + h1, x1:x1 + w1]

        x2 = int(f2[0])
        y2 = int(f2[1])
        w2 = int(f2[2])
        h2 = int(f2[3])
        img2 = img2[y2:y2 + h2, x2:x2 + w2]

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)

    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)

    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append([m])

    print(num2)
    print(len(good)/len(matches))

    # img3 = cv2.drawMatchesKnn(img1, keypoints_1, img2, keypoints_2, good, img2, flags=2)

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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    queryPath = "C:/Users/laub1/Desktop/4186/Queries/"
    comparePath = "C:/Users/laub1/Desktop/4186/Images/"
    Hist(queryPath + '01', comparePath + '0017')
    # for i in range(1, 2000):
    #     image = "0000" + str(i)
    #     image = image[-4:]
    #     SIFT_BF("01", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
