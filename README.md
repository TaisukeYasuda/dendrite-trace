# Dendrite Trace

15-112 Term Project at Carnegie Mellon University (Fall 2015)

### Overview

This program is a tool for neuroscientists for manually and automatically tracing the three dimensional structure of dendrites, given a three dimensional stack of images (see the [demo](https://www.youtube.com/watch?v=Hpq4ji-KllA)). The manual tracing option provides a simple user interface that allows the user to incrementally plot points along the dendrite. As the user moves along the dendrite, the program zooms into the region in three dimensions. In addition, this manual option also provides the option for generating training data based on the user's actions. This option will automatically create labeled images of positive and negative connections (connected and disconnected), which can be used to train a classification algorithm. A trained classification algorithm comes with the program, created using scikit-learn, which can be harnessed in the auto trace option. The auto trace does not work perfectly, but shows great potential with improvement, particularly through the use of more data and powerful features. 

### Dependencies

This program runs primarily on Python 2. In addition, it relies on several popular modules such as wxPython, OpenCV, and scikit-learn. On an Ubuntu linux machine, these can be installed by running the following commands:
```
sudo apt-get install python-wxgtk2.8
sudo apt-get install python-pip
sudo pip install cv2
sudo pip install numpy
sudo pip install scipy 
sudo pip install sklearn
sudo pip install pillow
```

### User Guide

First, clone the project from github to a directory of your choice. Then, navigate to the project folder directory with the command `cd ./DendriteTrace/dendrite_trace_0.0`. Then, the program is started by running the command `python DendriteTrace.py`. Once the program is started, the console on the upper right hand side of the GUI will prompt you for the next step in the process and tell you everything you need to know! Due to the technical specificity of the project, I have included several sample images in the folder `~/DendriteTrace/dendrite_trace_0.0/sample_images`. The general flow of the program is as follows:

* Open an image (TIF file)
* Select an initial point
* Select either manual mode or auto mode
  * Manual mode
    * Use the left and right keys to select the next points along the dendrite
    * Press space to enter points
    * Press enter to move to the next point
  * Auto mode
    * Press enter to start auto trace
    * Press fix to review program's work and add missing branches
    * Press add to add branches
    * Press trace to let the program auto trace from the added point
