#!/usr/bin/env python

# Copyright (C) Alex Ge, alexgecontrol@qq.com.
#
# This file is part of Robotic Arm: Pick and Place project for Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Alex Ge
import pcl
import numpy as np

from pcl_helper import XYZRGB_to_XYZ

class PlaneSegmenter():
    """ Dominant plane segmenter for PCL
    """
    def __init__(self, cloud, max_distance = 1):
        """ Instantiate plane segmenter
        """
        self._max_distance = max_distance

        self._segmenter = cloud.make_segmenter()
        self._segmenter.set_model_type(pcl.SACMODEL_PLANE)
        self._segmenter.set_method_type(pcl.SAC_RANSAC)
        self._segmenter.set_distance_threshold(self._max_distance)

    def segment(self):
        """ Segment the dominant plane
        """
        return self._segmenter.segment()

class EuclideanSegmenter():
    """ Segment PCL using DBSCAN
    """
    def __init__(
        self,
        cloud,
        eps = 0.001, min_samples = 10, max_samples = 250
    ):
        """ Instantiate Euclidean segmenter
        """
        # 1. Convert XYZRGB to XYZ:
        self._cloud = XYZRGB_to_XYZ(cloud)
        self._tree = self._cloud.make_kdtree()

        # 2. Set params:
        self._eps = eps
        self._min_samples = min_samples
        self._max_samples = max_samples

        # 3. Create segmenter:
        self._segmenter = self._cloud.make_EuclideanClusterExtraction()
        self._segmenter.set_ClusterTolerance(self._eps)
        self._segmenter.set_MinClusterSize(self._min_samples)
        self._segmenter.set_MaxClusterSize(self._max_samples)
        self._segmenter.set_SearchMethod(self._tree)

    def segment(self):
        """ Segment objects
        """
        # 1. Segment objects:
        cluster_indices = self._segmenter.Extract()

        # 2. Generate representative point for object:
        cluster_reps = []
        for idx_points in cluster_indices:
            object_cloud = self._cloud.extract(idx_points)

            # Use centroid as representative point:
            rep_position = np.mean(
                object_cloud.to_array(),
                axis=0
            )[:3]

            cluster_reps.append(rep_position)

        return (cluster_indices, cluster_reps)
