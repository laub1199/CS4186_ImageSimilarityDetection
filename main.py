from __future__ import division
import cv2
import numpy as np
from matplotlib import pyplot as plt

EXAMPLE_QUERY_PATH = './example_query/'
QUERY_PATH = './Queries/'
IMAGE_PATH = './Images/'

def SIFT(query, compare):
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(query, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(compare, None)

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    quality_match = []
    for m, n in matches:
        if m.distance < n.distance * 0.8:
            quality_match.append([m])
    return (len(quality_match)/len(matches))

def MSE(query, compare):
    # resize both image to 500 * 500
    query = cv2.resize(query, (500, 500), interpolation=cv2.INTER_AREA)
    compare = cv2.resize(compare, (500, 500), interpolation=cv2.INTER_AREA)

    # mse calculation
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

def genRankList(method, isExample):
    prefix = 'example_' if isExample else ''
    text_file = open(prefix + method + "_rankList.txt", "w")
    data_file = open(prefix + method + "_similarity_score.txt", "w")
    queryPath = EXAMPLE_QUERY_PATH if isExample else QUERY_PATH
    comparePath = IMAGE_PATH
    list = []

    q_to = 11 if isExample else 21

    for q in range(1, q_to):
        # get query's num from 01 to 20
        queryNum = "00" + str(q)
        queryNum = queryNum[-2:]

        query = cv2.imread(queryPath + queryNum + ".jpg")

        # get data from txt for image cropping
        f1 = open(queryPath + queryNum + ".txt").read().split()

        x1 = int(f1[0])
        y1 = int(f1[1])
        w1 = int(f1[2])
        h1 = int(f1[3])
        query = query[y1:y1 + h1, x1:x1 + w1]

        for i in range(1, 5001):
            # get compare image's num from 0001 to 5000
            compareNum = "0000" + str(i)
            compareNum = compareNum[-4:]
            similarity = None

            compare = cv2.imread(comparePath + compareNum + ".jpg")

            if int(str(comparePath + compareNum).split('Images/')[1]) <= 2000:
                f2 = open(comparePath + compareNum + ".txt").read().split()

                x2 = int(f2[0])
                y2 = int(f2[1])
                w2 = int(f2[2])
                h2 = int(f2[3])
                compare = compare[y2:y2 + h2, x2:x2 + w2]

            if method == "SIFT":
                similarity = SIFT(query, compare)
            if method == 'MSE':
                similarity = MSE(query, compare)

            print(queryNum + '-' + compareNum + ': ' + str(similarity))
            temp = [i, similarity]
            list.append(temp)
            data_file.write(str(q) + ' ' + compareNum + ' ' + str(similarity) + '\n')


        # reverse true = descending
        reverse = True if method == 'SIFT' else False
        list = sorted(list, key=lambda x: x[1], reverse=reverse)

        text_file.write('Q' + str(q) + ': ')
        for i in range(0, len(list)):
            text_file.write(str(list[i][0]))
            text_file.write(' ')

        text_file.write('\n')
        list = []

    data_file.close()
    text_file.close()

def method_combine (isExample):
    m1_path = './example_SIFT_similarity_score.txt' if isExample else './SIFT_similarity_score.txt'
    m2_path = './example_MSE_similarity_score.txt' if isExample else './MSE_similarity_score.txt'

    sim_score_path = './example_METHOD_COMBINE_similarity_score.txt' if isExample else './METHOD_COMBINE_similarity_score.txt'
    rankList_path = './example_METHOD_COMBINE_rankList.txt' if isExample else './METHOD_COMBINE_rankList.txt'

    file_m1 = open(m1_path).read().splitlines()
    file_m2 = open(m2_path).read().splitlines()
    out = open(sim_score_path, "w")
    text_file = open(rankList_path, "w")

    q_to = 10 if isExample else 20

    m2_list = []
    # get min max mean range of each query's MSE
    for x in range(0, q_to):
        max = 0
        min = 0
        mean = 0
        for y in range(0, 5000):
            prefix = x * 5000
            line = prefix + y
            m2_sim = float(file_m2[line].split(' ')[2])
            max = m2_sim if m2_sim > max else max
            min = m2_sim if m2_sim < min else min
            mean = mean + m2_sim

        mean = mean / 5000
        temp=[min, mean, max, max-min]
        m2_list.append(temp)

    for x in range(0, q_to):
        list = []
        for y in range(0, 5000):
            prefix = x * 5000
            line = prefix + y
            m1_sim = 1000000 * float(file_m1[line].split(' ')[2])
            m2_sim = float(file_m2[line].split(' ')[2])
            accuracy_score = 1 - (m2_sim - m2_list[x][0]) / (m2_list[x][3])
            sim = m1_sim * accuracy_score
            out.write(str(file_m1[line].split(' ')[0]) + ' ' + file_m1[line].split(' ')[1] + ' ' + str(sim) + '\n')
            temp=[y+1, sim]
            list.append(temp)

        # reverse true = descending
        list = sorted(list, key=lambda x: x[1], reverse=True)

        text_file.write('Q' + str(x+1) + ': ')
        for i in range(0, len(list)):
            text_file.write(str(list[i][0]))
            text_file.write(' ')
        text_file.write('\n')
    print('combined')
    out.close()
    text_file.close()

def top_10_image (isExample):
    path_prefix = './example_' if isExample else './'
    methods = ['SIFT_', 'MSE_', 'METHOD_COMBINE_']
    rankList_path = 'rankList.txt'

    q_to = 10 if isExample else 20

    for method in methods:
        path = path_prefix + method + rankList_path
        rankList = open(path).read().splitlines()
        fig = plt.figure(figsize=(128, 128))

        for x in range(0, q_to):
            for y in range(1, 11):
                num = rankList[x].split(' ')[y]

                image_name = "0000" + str(num)
                image_name = image_name[-4:]

                img = cv2.imread('./Images/' + image_name + '.jpg')
                fig.add_subplot(q_to, 10, x*10+y)
                plt.axis('off')
                plt.imshow(img)
        plt.savefig(path_prefix + method + "top_10.png")
        print('saved')






if __name__ == '__main__':
    isExample = True
    genRankList('MSE', isExample)
    genRankList('SIFT', isExample)
    method_combine(isExample)
    top_10_image(isExample)

    isExample = False
    genRankList('MSE', isExample)
    genRankList('SIFT', isExample)
    method_combine(isExample)
    top_10_image(isExample)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/