
########################## Dendrite Trace Image ##########################
# This is a file that contains static functions for Dendrite Trace dealing
# with images.
##########################################################################

import math
import copy
import wx
import cv2
import numpy
import scipy
import scipy.misc
from DendriteTrace import DendriteTrace
from DendriteTracePoint import DendriteTracePoint as dt_point
from PIL import Image

# given a path to a three dimensional tiff image stack, this function 
# returns the corresponding three dimensional numpy array
def load_image(path):
    img = Image.open(path)
    # determine the number of frames in the image
    num_frames = 0
    try:
        while True:
            img.seek(num_frames)
            num_frames += 1
    except:
        img.seek(0)
    # copy each frame into a numpy array
    img_array = numpy.zeros((img.size[0],img.size[1],num_frames),numpy.uint16)
    frame = 0
    try:
        while True:
            img.seek(frame)
            img_array[:,:,frame] = img
            frame = frame + 1
    except:
        img.seek(0)
    # returns three dimensional numpy array of the image
    return img_array


# given a three dimensional numpy array image, this function returns the 
# image projected down along the z axis
def proj_image(img):
    # flatten the dendrite image by taking the max along the depth axis
    img = numpy.ndarray.max(img,axis=2)
    return img


# given a numpy image, this function returns the equivalent wxbitmap
def numpy_to_wxbitmap(img):
    # save as tif file first, then read as wx.Bitmap type
    scipy.misc.imsave(DendriteTrace.IMAGE_PATH,img)
    return wx.Bitmap(DendriteTrace.IMAGE_PATH,wx.BITMAP_TYPE_TIF)


# given a three dimensional numpy image, this function filters the image and 
# subtracts the background intensity
def process_image(img):
    # filter each image stack slice
    filter_size = (3,3)
    size = img.shape
    depth = size[2]
    for i in range(depth):
        img[:,:,i] = cv2.blur(img[:,:,i].astype('uint16'),filter_size)

    # calculate background intensity
    hist = cv2.calcHist([img[0]],[0],None,[65535],[0,65535])
    background = hist.argmax()
    img = img - background*numpy.ones(size)
    negative = (img < 0)
    img = numpy.maximum(img,negative)
    return img

# the region of interest when you zoom in
def roi(parameters,point):
    x,y,z = point.point
    r = parameters['zoom_size']/2
    height = width = parameters['max_image_size']

    # check for out of bounds in x and y
    bot,top,left,right = y>height-r,y<r,x<r,x>width-r
 
    if(bot):
        bot_bounds,bot_edge = height-y+r,r+(height-y)
        if(left):
            left_bounds,right_bounds = 0,x+r
            left_edge,right_edge = r-x,2*r
        elif(right):
            left_bounds,right_bounds = x-r,width
            left_edge,right_edge = 0,r+(width-x)
        else:
            left_bounds,right_bounds = x-r,x+r
            left_edge,right_edge = 0,2*r
        bot_bounds,top_bounds = height-bot_bounds,height
        bot_edge,top_edge = 0,bot_edge
    elif(top):
        top_bounds,top_edge = y+r,y+r
        if(left):
            left_bounds,right_bounds = 0,x+r
            left_edge,right_edge = r-x,2*r
        elif(right):
            left_bounds,right_bounds = x-r,width
            left_edge,right_edge = 0,r+(width-x)
        else:
            left_bounds,right_bounds = x-r,x+r
            left_edge,right_edge = 0,2*r
        bot_bounds,top_bounds = 0,top_bounds
        bot_edge,top_edge = 2*r-top_edge,2*r
    else:
        if(left):
            left_bounds,right_bounds = 0,x+r
            left_edge,right_edge = r-x,2*r
        elif(right):
            left_bounds,right_bounds = x-r,width
            left_edge,right_edge = 0,r+(width-x)
        else:
            left_bounds,right_bounds = x-r,x+r
            left_edge,right_edge = 0,2*r
        bot_bounds,top_bounds = y-r,y+r
        bot_edge,top_edge = 0,2*r

    bot_bounds,top_bounds = round(bot_bounds),round(top_bounds)
    left_bounds,right_bounds = round(left_bounds),round(right_bounds)
    bot_edge,top_edge = round(bot_edge),round(top_edge)
    left_edge,right_edge = round(left_edge),round(right_edge)

    edges = (bot_edge,top_edge,left_edge,right_edge)
    bounds = (bot_bounds,top_bounds,left_bounds,right_bounds)

    summary = dict()
    summary['edges'] = edges
    summary['bounds'] = bounds
    return summary


def zoom_2d(self):
    # display the zoom image
    parameters = self.parameters
    summary = roi(parameters,self.curr_point)
    # take only the layer plus and minus the zoom_depth
    zoom_size = parameters['zoom_size']
    max_image_size = parameters['max_image_size']
    size = (zoom_size,zoom_size)
    self.zoom_img = numpy.zeros(size)
    bot_edge,top_edge,left_edge,right_edge = summary['edges']
    bot_bounds,top_bounds,left_bounds,right_bounds = summary['bounds']

    section = self.proj_img[bot_bounds:top_bounds,left_bounds:right_bounds]
    self.zoom_img[bot_edge:top_edge,left_edge:right_edge] = section

    size = (max_image_size,max_image_size)
    zoom_img = scipy.misc.imresize(self.zoom_img,size)
    return numpy_to_wxbitmap(zoom_img)


def zoom_3d(self):
    parameters = self.parameters
    max_image_size = parameters['max_image_size']
    size = (max_image_size,max_image_size)
    self.zoom_img = scipy.misc.imresize(zoom(self),size)
    return numpy_to_wxbitmap(self.zoom_img)


# similar zoom_3d but get to specify points and returns 2d numpy array
def zoom(self,point=None):
    # find the region of interest
    parameters = self.parameters
    if(point==None):
        point = copy.copy(self.curr_point)
    x,y,z = point.point
    summary = roi(parameters,self.curr_point)
    bot_edge,top_edge,left_edge,right_edge = summary['edges']
    bot_bounds,top_bounds,left_bounds,right_bounds = summary['bounds']

    # take only the layer plus and minus the zoom_depth
    zoom_size = parameters['zoom_size']
    zoom_depth = parameters['zoom_depth']
    size = (zoom_size,zoom_size,zoom_depth*2+1)
    zoom_img = numpy.zeros(size)

    img = self.img
    size = img.shape
    for i in range(-zoom_depth,zoom_depth+1):
        index = i + z
        if(index >= 0 and index < size[2]):
            section = img[bot_bounds:top_bounds,
                left_bounds:right_bounds,index]
            zoom_img[bot_edge:top_edge,left_edge:right_edge,i] = section

    return proj_image(zoom_img)



def format_data(self,img,w,h,size,theta,zoom_size):
    # rotate, crop, and resize the image
    theta = 90 - theta # rotate so that center of roi is at 90 degrees
    img = scipy.misc.imrotate(img,theta)
    r = zoom_size/2
    img = img[r-h:r,r-w/2:r+w/2]
    img = cv2.resize(img,size)
    return img


# takes numpy arrays
def value(self,img,theta):
    parameters = self.parameters
    w = parameters['value_width']
    h = parameters['value_height']
    s = parameters['value_size']
    z_s = parameters['zoom_size']
    return format_data(self,img,w,h,s,theta,z_s)


# takes numpy arrays
def reward(self,img,theta):
    parameters = self.parameters
    w = parameters['reward_width']
    h = parameters['reward_height']
    s = parameters['reward_size']
    z_s = parameters['zoom_size']
    return format_data(self,img,w,h,s,theta,z_s)


# takes numpy arrays
def training_data(self,img,theta):
    parameters = self.parameters
    w = parameters['reward_width']
    h = parameters['manual_radius']
    s = parameters['reward_size']
    z_s = parameters['zoom_size']
    return format_data(self,img,w,h,s,theta,z_s)


# takes a predict list and finds the midpoint angles
def find_midpoints(predict_list):
    predict_list = numpy.asarray(predict_list)
    predict_list = predict_list.flatten()
    length = len(predict_list)
    unit_theta = 360 / length
    connections = list()
    connection = list()
    midpoints = list()
    # store connected groups
    for i in range(length):
        if(predict_list[i]==1):
            connection.append(i)
        else:
            if(len(connection)>0):
                connections.append(connection)
            connection = list()
    if(len(connection)>0):
        connections.append(connection)
    # combine first and last connections if necessary
    if(len(connections)==0):
        return list()
    length_1 = len(connections) - 1
    length_2 = len(connections[length_1]) - 1
    first = connections[0][0]
    last = connections[length_1][length_2]
    if(first==0 and last==length-1):
        for i in range(len(connections[0])):
            connections[0][i] += length
        connections[length_1].extend(connections.pop(0))
    for connection in connections:
        length = len(connection)
        if(length > 0):
            midpoint = (connection[0] + connection[length-1]) / 2.0 * unit_theta
            midpoint = midpoint % 360
            midpoints.append(midpoint)
    return midpoints