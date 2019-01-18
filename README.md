Template repository for generating training data for supervised learning algorithms working on occupancy grid maps.
## Summary
This is a simulator for generating occupancy grid maps of the indoor environment automatically. From the simulated maps, the image patches containing a doorway or background can be extracted and besides the mask of the doorway can be annotated. The simulated data can be used as training data for supervised deep learning methods.

## Requirments:
* Python3
* numpy
* scipy
* OpenCV-Python
* Scikit-Image
* imutils

## Folders:
### 1. Samples:
Generated maps with different types of noise  
* __Samples/map_noNoise.png__: simulated map of an indoor environment with some measurement erros, but without noise
* __Samples/map_s&pNoise.png__: simulated map of an indoor environment with some measurement erros and salt&pepper noise
* __Samples/map_combNoise.png__: simulated map of an indoor environment with some measurement erros and combined noise
* __Samples/map_GaussNoise.png__: simulated map of an indoor environment with some measurement erros and Gaussian noise

### 2. Code:

* __Code/config.py__: contains the tunable parameters of the simulator
* __Code/utils.py__: contains the common utility functions
* __Code/trunk.py__: contains a class for creating the trunk part of the map, including the corridor, the pillar inside the doorway, and doors.
* __Code/addObjects.py__: contains a class for adding grooves on the wall and furnitures to the room.
* __Code/orig_map.py__: contains a class for creating the original map without noise and only in the vertical direction.
* __Code/addNoise.py__: contains a class for adding different types of noise to the original map. The available options include 'noNoise', 'spNoise', 'combindNoise', 'GaussNoise'.
* __Code/dataExtraction.py__: contains a class for extracting data and its mask from simulated maps
* __Code/main.py__: contains a function to run the simulator

## How to use:
The parameters in __Code/config.py__ can be tuned according to the specific requirments. Please use __Code/main.py__ to run the simulator.
The required inputs are
* map_num: the number of simulated maps
* noise_types: the types of added noise ('noNoise','spNoise','combindNoise','GaussNoise')
* noise_levels: the levels of added noise (0,1,2,...)
* mode: 0 for extracting patches of background, 1 for extracting patches containing a doorway
=======
# SemanticMapGeneration
Template repository for generating training data for supervised learning algorithms working on occupancy grid maps.
>>>>>>> 9925dda8ca4c3d18bb01ddfadbce6cba0b446a08
