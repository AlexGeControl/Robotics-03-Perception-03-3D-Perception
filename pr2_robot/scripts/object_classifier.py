#!/usr/bin/env python

# Set up session:
import pickle

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from rospy_message_converter import message_converter
import yaml

from pr2_robot.srv import *
from pr2_robot.msg import *
from pcl_processing.pcl_helper import *
from pcl_processing.pcl_filter import *
from pcl_processing.pcl_segmenter import *
from pcl_processing.features import compute_color_histograms
from pcl_processing.features import compute_normal_histograms
from pcl_processing.marker_tools import *
from visualization_msgs.msg import Marker

from sklearn.preprocessing import LabelEncoder

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/pr2_feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

class PR2Mover():
    """ Segmented PCL classifier
    """
    def __init__(self, model_filename):
        """ Initialize PR2 mover node
        """
        # Load trained classifier:
        model = pickle.load(
            open(model_filename, 'rb')
        )

        # 1. Classifier:
        self._clf = model['classifier']
        # 2. Label encoder:
        self._encoder = LabelEncoder()
        self._encoder.classes_ = model['classes']
        # 3. Feature scaler:
        self._scaler = model['scaler']

        # Initialize ros node:
        rospy.init_node('pr2_mover')

        # Create Subscribers
        self._sub_pcl = rospy.Subscriber(
            '/pr2/world/points',
            PointCloud2,
            self._handle_pcl_classification,
            queue_size=10
        )

        # Create Publishers
        self._pub_pcl_processed = rospy.Publisher(
            '/pr2/roi/points',
            PointCloud2,
            queue_size=10
        )
        self._pub_pcl_objects = rospy.Publisher(
            '/detected_objects',
            DetectedObjectsArray,
            queue_size=10
        )
        self._pub_pcl_labels = rospy.Publisher(
            '/object_markers',
            Marker,
            queue_size=10
        )

        # Spin till shutdown:
        while not rospy.is_shutdown():
            rospy.spin()

    def _handle_pcl_classification(self, ros_cloud):
        """ Handle ROS pc2 message
        """
        # Convert ROS msg to PCL data
        pcl_original = ros_to_pcl(ros_cloud)

        # 1. Voxel grid downsampling
        downsampler = VoxelFilter(
            pcl_original,
            0.005
        )
        pcl_downsampled = downsampler.filter()

        # 2. PassThrough filter
        pcl_roi = pcl_downsampled
        for axis_name, axis_range in zip(
            ('z', 'y', 'x'),
            (
                [+0.60, +1.20],
                [-0.50, +0.50],
                [+0.33, +0.90],
            )
        ):
            roi_filter = PassThroughFilter(
                pcl_roi,
                axis_name,
                axis_range
            )
            pcl_roi = roi_filter.filter()

        # 3. Outlier filter:
        outlier_filter = OutlierFilter(
            pcl_roi,
            k = 50,
            factor = 1
        )
        pcl_denoised = outlier_filter.filter()

        # 4. RANSAC plane segmentation
        plane_segmenter = PlaneSegmenter(
            pcl_denoised,
            0.005
        )
        (idx_table, normal_table) = plane_segmenter.segment()

        # 4. Extract objects:
        pcl_objects = pcl_denoised.extract(idx_table, negative=True)

        # 5. Outlier filter:
        outlier_filter = OutlierFilter(
            pcl_objects,
            k = 25,
            factor = 1
        )
        pcl_objects = outlier_filter.filter()

        self._pub_pcl_processed.publish(
            pcl_to_ros(pcl_objects)
        )

        # 6. Extract seperate objects using DBSCAN
        object_segmenter = EuclideanSegmenter(
            pcl_objects,
            eps = 0.05, min_samples = 32, max_samples = 4096
        )
        (cluster_indices, cluster_reps) = object_segmenter.segment()

        # 7. Detect objects:
        detected_objects = []
        detected_object_labels = []
        for idx_object, idx_points in enumerate(cluster_indices):
            # Grab the points for the cluster from the extracted outliers (cloud_objects)
            pcl_object = pcl_objects.extract(idx_points)

            # Convert the cluster from pcl to ROS using helper function
            ros_cloud_object = pcl_to_ros(pcl_object)

            # Extract histogram features
            color_hist = compute_color_histograms(ros_cloud_object, using_hsv=True)
            normal_hist = compute_normal_histograms(get_normals(ros_cloud_object))
            feature = np.concatenate(
                (color_hist, normal_hist)
            )

            # Make the prediction
            prediction = self._clf.predict(
                self._scaler.transform(
                    feature.reshape(1,-1)
                )
            )
            label = self._encoder.inverse_transform(
                prediction
            )[0]

            # Add the detected object to the list of detected objects.
            detected_object = DetectedObject()
            detected_object.label = label
            detected_object.cloud = pcl_object
            detected_objects.append(
                detected_object
            )
            detected_object_labels.append(label)
            # Publish object label into RViz
            self._pub_pcl_labels.publish(
                make_label(label,cluster_reps[idx_object], idx_object)
            )

        # 8. Prompt:
        rospy.loginfo(
            'Detected {} objects: {}'.format(
                len(detected_object_labels),
                detected_object_labels
            )
        )
        self._pub_pcl_objects.publish(detected_objects)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data

    # TODO: Statistical Outlier Filtering

    # TODO: Voxel Grid Downsampling

    # TODO: PassThrough Filter

    # TODO: RANSAC Plane Segmentation

    # TODO: Extract inliers and outliers

    # TODO: Euclidean Clustering

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately

    # TODO: Convert PCL data to ROS messages

    # TODO: Publish ROS messages

# Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)

        # Grab the points for the cluster

        # Compute the associated feature vector

        # Make the prediction

        # Publish a label into RViz

        # Add the detected object to the list of detected objects.

    # Publish the list of detected objects

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects_list)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables

    # TODO: Get/Read parameters

    # TODO: Parse parameters into individual variables

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list

        # TODO: Get the PointCloud for a given object and obtain it's centroid

        # TODO: Create 'place_pose' for the object

        # TODO: Assign the arm to be used for pick_place

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file



if __name__ == '__main__':
    PR2Mover('model.pkl')
