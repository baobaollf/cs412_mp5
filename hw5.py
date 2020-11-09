# Linfeng Li
# CS 415
# University of Illinois at Chicago
# 10/30/2020

import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import os
import math
from sklearn.cluster import KMeans


def read_images(count, folder):
    images = {}
    for filename in os.listdir(folder):
        local_count = 0
        path = folder + '/' + filename
        category = []
        for cat in os.listdir(path):
            if local_count < count:
                image_path = path + '/' + cat
                img = cv2.imread(image_path, 0)
                # plt.imshow(img)
                # plt.show()
                category.append(img)
                local_count += 1
        images[filename] = category
    return images


def sift_function(all_images):
    sift_vectors = {}
    descriptor_list = []
    for key, value in all_images.items():
        features = []
        for img in value:
            sift = cv2.SIFT_create()
            kp, des = sift.detectAndCompute(img, None)
            descriptor_list.extend(des)
            features.append(des)
        sift_vectors[key] = features
    return [descriptor_list, sift_vectors]


def kmeans(k, descriptors):
    descriptors = np.array(descriptors)
    ret, label, center = cv2.kmeans(data=descriptors, K=k,
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                    attempts=1,
                                    flags=cv2.KMEANS_RANDOM_CENTERS,
                                    bestLabels=None)
    return center


def find_min_index(sift_descriptor, center):
    distances = []
    for i in center:
        local_distance = math.dist(sift_descriptor, i)
        distances.append(local_distance)
    return np.argmin(distances)


def build_histogram(centers, all_bow):
    dict_feature = {}
    for key, value in all_bow.items():
        category = []
        for img in value:
            histogram = np.zeros(len(centers))
            for each_feature in img:
                index = find_min_index(each_feature, centers)
                histogram[index] += 1
            print(histogram)
            category.append(histogram)
        dict_feature[key] = category
    return dict_feature


def knn(images, tests):
    train_histograms = []
    train_labels = []
    test_histograms = []
    test_labels = []
    for category in images:
        for histogram in images[category]:
            train_histograms.append(histogram)
            train_labels.append(category)
    for category in tests:
        for histogram in tests[category]:
            test_histograms.append(histogram)
            test_labels.append(category)

    predictions = []
    for histogram in test_histograms:
        distance = []
        for train_histogram in train_histograms:
            local_distance = math.dist(histogram, train_histogram)
            distance.append(local_distance)
        predictions.append(train_labels[int(np.argmin(distance))])
    return predictions, test_labels


def find_accuracy(predictions, validation_true_label):
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == validation_true_label[i]:
            count += 1
    accuracy = count / len(predictions)
    print(accuracy)
    return accuracy


def main():
    validation_folder = './data/validation'
    train_folder = './data/train'
    plt.gray()
    np.set_printoptions(suppress=True)
    train_images = read_images(50, train_folder)
    test_images = read_images(50, validation_folder)
    sifts = sift_function(train_images)

    descriptor_list = sifts[0]
    train_bow_feature = sifts[1]

    test_bow_feature = sift_function(test_images)[1]

    center = kmeans(200, descriptor_list)

    bow_train = build_histogram(center, train_bow_feature)

    bow_test = build_histogram(center, test_bow_feature)

    predictions, true_label = knn(bow_train, bow_test)
    for x in range(len(predictions)):
        print(predictions[x], true_label[x])
    find_accuracy(predictions, true_label)

    # vote = np.zeros(10)
    # for i in label:
    #     vote[i] += 1
    # # print((vote))
    # knn(clean_list, label)

    # plt.imshow(all_images['TallBuilding'][0])
    # plt.show()
    # # image save path
    # gun1_save_path_1 = os.path.join("results/gun1.png")
    # joy1_save_path_2 = os.path.join("results/joy1.png")
    # pointer1_save_path_3 = os.path.join("results/pointer1.png")
    #
    # # read image
    # gun1_image = cv2.imread(gun1_path)
    # joy1_image = cv2.imread(joy1)
    # pointer1_image = cv2.imread(pointer1)
    #
    # # processed images
    # gun1_hsv = cv2.cvtColor(gun1_image, cv2.COLOR_BGR2HSV)
    # joy1_hsv = cv2.cvtColor(joy1_image, cv2.COLOR_BGR2HSV)
    # pointer_hsv = cv2.cvtColor(pointer1_image, cv2.COLOR_BGR2HSV)
    # matrix = histogram()
    # plot_histogram(matrix)
    #
    # # save images
    # plt.imsave(gun1_save_path_1, show_results(gun1_hsv, matrix))
    # plt.imsave(joy1_save_path_2, show_results(joy1_hsv, matrix))
    # plt.imsave(pointer1_save_path_3, show_results(pointer_hsv, matrix))
    return


if __name__ == '__main__':
    main()
