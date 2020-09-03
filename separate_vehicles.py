#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:16:08 2020

@author: heldhy
"""

import sys
import numpy as np
import cv2 as cv

from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage import morphology
from skimage.segmentation import watershed, mark_boundaries
from skimage.feature import peak_local_max
from skimage.color import rgb2gray

WHITE_VALUE = 255


def noise_removal(img):
    denoised_image = morphology.opening(img, morphology.square(3))
    return denoised_image


def separate_vehicles_by_size(denoised_img, separation_size):
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(denoised_img, connectivity=8)
    #dropping background
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    #result images
    unique_vehicles = np.zeros((output.shape))
    clustered_vehicles = np.zeros((output.shape))
    for component_id in range(nb_components):
        if (sizes[component_id] >= separation_size):
            clustered_vehicles[output == component_id + 1] = WHITE_VALUE
        else:
            unique_vehicles[output == component_id + 1] = WHITE_VALUE
    single_vehicle_sizes = sizes[sizes < separation_size]
    single_vehicle_mean_size = int(np.sum(single_vehicle_sizes) / len(single_vehicle_sizes))
    return np.uint8(unique_vehicles), np.uint8(clustered_vehicles), single_vehicle_mean_size


def is_connected_to(image, label_of_interest, other_label):
    merged_labels = np.uint8(np.zeros(image.shape))
    merged_labels[image == label_of_interest] = 255
    merged_labels[image == other_label] = 255
    nb_components = cv.connectedComponents(merged_labels, connectivity=8)[0]
    #One component for the background + one for the connected labels
    return nb_components == 2


def connected_labels(image, label_of_interest):
    connected_labels = []
    for current_label in range(1, np.max(image) + 1):
        if(current_label != label_of_interest and is_connected_to(image, label_of_interest, current_label)):
            connected_labels.append(current_label)
    return connected_labels


def reassignate_cluster_number(image, first_cluster_number=1):
    new_label_count = first_cluster_number
    new_labels = np.uint8(np.zeros(image.shape))
    for current_label in range(1, np.max(image) + 1):
        if(np.count_nonzero(image == current_label) > 0):
            new_labels[image == current_label] = new_label_count
            new_label_count += 1
    return new_labels


def merge_labels(image, min_size = 80):
    number_of_labels_before_merging = len(np.unique(image))
    for current_label in range(np.max(image) + 1):
        cluster_size = np.count_nonzero(image == current_label)
        if(cluster_size > 0 and cluster_size < min_size):
            label_connected = connected_labels(image, current_label)
            if (label_connected != []):
                label_sizes = [np.count_nonzero(image == possible_label) for possible_label in label_connected]
                smallest_connected_label = label_connected[np.argmin(label_sizes)]
                image[image == current_label] = smallest_connected_label
                break
    number_of_labels_after_merging = len(np.unique(image))
    new_image = reassignate_cluster_number(image)
    if (number_of_labels_before_merging > number_of_labels_after_merging):
        image = merge_labels(new_image, min_size)
    return image


def extract_single_vehicles(base_img, unique_vehicles):
    markers = ndi.label(unique_vehicles)[0]
    nb_vehicles = len(np.unique(markers)) - 1
    img = mark_boundaries(base_img, markers, color=[0,255,0], mode="outer")
    return img, markers, nb_vehicles


def extract_clustered_vehicles(base_img, clustered_vehicles, single_vehicle_mean_size):
    distance = ndi.distance_transform_edt(clustered_vehicles)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), \
                                labels=clustered_vehicles,                          \
                                num_peaks_per_label=single_vehicle_mean_size)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=clustered_vehicles)
    watershed_markers = merge_labels(labels.copy())
        
    nb_vehicles = len(np.unique(watershed_markers)) - 1
    img = mark_boundaries(base_img, watershed_markers, color=[255,0,0], mode="outer")
    return img, watershed_markers, nb_vehicles


def merging_single_and_clustered_results(base_img, unique_markers, clustered_markers, nb_single_vehicles):
    new_clustered_marker = reassignate_cluster_number(clustered_markers, nb_single_vehicles + 1)
    merged_markers = new_clustered_marker + unique_markers
    img = mark_boundaries(base_img, merged_markers, color=[255,0,0], mode="outer")
    nb_vehicles = len(np.unique(merged_markers)) - 1
    return img, merged_markers, nb_vehicles


def main():
    TERMINAL_MODE = True #if launched from an IDE, change TERMINAL MODE to False
    if (TERMINAL_MODE and len(sys.argv) < 3):
        print("USAGE:\n")
        print("python3 separate_vehicles.py [PREDICTION PATH] [IMAGE TO DISPLAY THE SEGMENTATION ON PATH]")
        print("Results will be saved as segmented_image.png")
    else:
        if (TERMINAL_MODE):
            img = plt.imread(sys.argv[1])
            background_image = plt.imread(sys.argv[2])
        else:
            img = plt.imread("prediction_cars.tif")
            background_image = plt.imread("DG_2016_02_20.tif")
        gray = rgb2gray(img)
        denoised_image = noise_removal(gray)
        unique_vehicles, clustered_vehicles, single_vehicle_mean_size = separate_vehicles_by_size(denoised_image, separation_size = 160)
        unique_vehicles_results, unique_markers, nb_single_vehicles = extract_single_vehicles(img, unique_vehicles)
        clustered_vehicles_results, clustered_markers, nb_clustered_vehicles = extract_clustered_vehicles(img, clustered_vehicles, single_vehicle_mean_size)
        segmented_vehicles_results, markers, nb_vehicles = merging_single_and_clustered_results(background_image, unique_markers, clustered_markers, nb_single_vehicles)
        print("number of detected vehicles "+ str(nb_vehicles))
        plt.figure(figsize=(20,20))
        plt.imshow(segmented_vehicles_results)
        plt.imsave("segmented_image.png", segmented_vehicles_results)


if __name__ == "__main__":
    main()