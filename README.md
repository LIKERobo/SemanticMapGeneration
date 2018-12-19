
## Summary
This is a simulator for generating occupancy grid maps of the indoor environment automatically. From the simulated maps, the image patches containing a doorway or background can be extracted and besides the mask of the doorway can be annotated. The simulated data can be used as training data for supervised deep learning methods.

## Modules:
### 1. auto_summit:
Package for controlling the robot SUMMIT XL. (usable for simulation and the real robot)   
    Important files: 
* __launch/summit_teleop.launch__: launchfile for controlling the robot SUMMIT XL by the keyboard (based on the launch file of the turtlebot)

### 2. obstacle_detector:
Package with the developed algorithm (based on the package [obstacle_detector](https://github.com/tysik/obstacle_detector) of Mateusz Przybyla). It contains the launch files for starting the algorithm, the C++-files with the algorithm and some python scripts for testing and evaluation of the modules of the algorithm.   
1. Important launchfiles:
* __obstacle_detector/launch/gazebo_detection.launch__: launchfile for starting the algorithm in the simulation environment gazebo
* __obstacle_detector/launch/robolab_detection.launch__: launchfile for starting the algorithm at the real robot
    
2. Important C++ files:
* __obstacle_detector/include/obstacle_detector/obstacle_extractor.h__: Contains the declarations of the variables and functions of the object detection algorithm 
* __obstacle_detector/src/obstacle_extractor.cpp__: Contains the implementation of the object detection algorithm 
* __obstacle_detector/include/obstacle_detector/obstacle_tracker.h__: Contains the declarations of the variables and functions of the object association algorithm 
* __obstacle_detector/src/obstacle_tracker.cpp__: Contains the implementation of the object association algorithm 
* __obstacle_detector/include/obstacle_detector/utilities/object.h__: File for description of a detected object with parts of the algorithms of object detection, classification and association
    
3. Important script files:
* __obstacle_detector/scripts/circle_detection.py__: Python script for testing and evaluation of circle detection algorithms (Hough Transform and Least Squares Regression)
* __obstacle_detector/scripts/circle_center_distribution.py__: Python script for testing and evaluation of position estimation accuracy of circle detection algorithms (Hough Transform and Least Squares Regression)
* __obstacle_detector/scripts/circle_classification_evaluation.py__: Python script for testing and evaluation of circle classification by Logistic Regression
* __obstacle_detector/scripts/test_circle_features.py__: Python script for testing and evaluation of features for classification of circles
* __obstacle_detector/scripts/circle_point_set_fusion.py__: Python script for testing and evaluation of point set fusion (point cloud simplification) algorithms (recursive k-Means, Quadtree and mean point calculation method)
* __obstacle_detector/scripts/circle_plot_pointsets.py__: Python script for testing and evaluation of a scan matching and two segmentation algorithms (ROS-Node)
* __obstacle_detector/scripts/rectangle_detection.py__: Python script for testing and evaluation of rectangle detection algorithms (Hough Transform, Least Squares Regression and Search Based Rectangle Fitting)
* __obstacle_detector/scripts/rectangle_center_distribution.py__: Python script for testing and evaluation of position estimation accuracy of rectangle detection algorithms (Hough Transform, Least Squares Regression and Search Based Rectangle Fitting)
* __obstacle_detector/scripts/rectangle_classification_evaluation.py__: Python script for testing and evaluation of rectangle classification by Logistic Regression
* __obstacle_detector/scripts/test_rectangle_features.py__: Python script for testing and evaluation of features for classification of rectangles