#!/usr/bin/env python

# Copyright (C) Alex Ge, alexgecontrol@qq.com.
#
# This file is part of Robotic Arm: Pick and Place project for Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Alex Ge

class VoxelFilter():
    """ Voxel filter for point cloud
    """
    def __init__(self, cloud, leaf_size = 0.0618):
        """ Instantiate voxel filter

        Args:
                cloud: pcl.cloud, input point cloud
            leaf_size: voxel dimension

        Returns:

        """
        self._leaf_size = leaf_size

        self._filter = cloud.make_voxel_grid_filter()
        self._filter.set_leaf_size(
            *([self._leaf_size]*3)
        )

    def filter(self):
        """ Filter input point cloud
        """
        return self._filter.filter()

class PassThroughFilter():
    """ Pass through filter for spatial ROI selection
    """
    def __init__(
        self,
        cloud,
        name,
        limits
    ):
        """ Instantiate pass through filter

        Args:
                cloud: pcl.cloud, input point cloud
                 name: filter field name
               limits: filter field value range, in (min, max) format
        """
        self._name = name
        self._limits = limits

        self._filter = cloud.make_passthrough_filter()
        self._filter.set_filter_field_name(
            self._name
        )
        self._filter.set_filter_limits(
            *self._limits
        )

    def filter(self):
        """ Filter input point cloud
        """
        return self._filter.filter()

class OutlierFilter():
    """ Remove outliers in PCL
    """
    def __init__(self, cloud, k = 50, factor = 1):
        """ Instantiate statistical outlier filter
        """
        self._k = k
        self._factor = factor

        self._filter = cloud.make_statistical_outlier_filter()
        # Set the number of neighboring points to analyze for any given point
        self._filter.set_mean_k(self._k)
        # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
        self._filter.set_std_dev_mul_thresh(self._factor)

    def filter(self):
        """ Return inliers
        """
        return self._filter.filter()
