
########################## Dendrite Trace Point #*#########################
# This is a file that contains the DendriteTracePoint class that keeps
# track of dendrite points and connections between them. It makes
# the management of the points much more efficient.
###########################################################################

import wx
import math
import copy
import numpy
import DendriteTraceImage as dt_image

class DendriteTracePoint(object):
    RADIUS = 3
    ZOOM_RADIUS = 6

    def __init__(self,point,center=(0,0,0)):
        # properties of point when not center
        self.point = point              # tuple for absolute coordinates
        self.center = center            # tuple center of zoom_image
        self.theta = None               # angle in degrees from center
        self.draw_point = None          # tuple for drawing coordinates

        # properties of point when center
        self.connections = list()       # list for adjacent points
        self.theta_list = list()        # list for angles of adjacent points
        self.bad_theta_list = list()    # list for angles for negative data

        self.calc_theta()
        self.calc_draw_point()

    def calc_theta(self):
        x,y,z = self.point
        x_c,y_c,z_c = self.center
        self.theta = math.atan2(-(y-y_c),x-x_c)*180/math.pi

    def calc_bad_theta(self):
        self.bad_theta_list = list()
        theta_list = copy.copy(self.theta_list)
        theta_list.sort()
        theta_naught = theta_list[0]
        while(len(theta_list) > 1):
            theta_0 = theta_list[0]
            theta_1 = theta_list[1]
            avg = (theta_0 + theta_1)/2
            self.bad_theta_list.append(avg)
            theta_list = theta_list[1:]
        avg = (theta_naught + theta_list.pop())/2 + 180
        self.bad_theta_list.append(avg)

    def calc_draw_point(self):
        x,y,z = self.point
        r = DendriteTracePoint.RADIUS
        self.draw_point = (x-r,y-r,2*r,2*r)

    def add_connection(self,conn):
        if(conn not in self.connections):
            curr_x,curr_y,curr_z = self.point
            next_x,next_y,next_z = conn.point
            self.connections += [conn]
            theta = math.atan2(-(next_y-curr_y),(next_x-curr_x))*180/math.pi
            self.theta_list += [theta]

    def remove_connection(self):
        self.connections.pop()
        self.theta_list.pop()

    # method for drawing on zoom_image
    def draw(self, curr_point, parameters, canvas):
        x,y,z = self.point
        x_c,y_c,z_c = curr_point.point
        r = parameters['zoom_size'] / 2
        ratio = parameters['max_image_size'] / parameters['zoom_size']
        if(abs(x-x_c)<r and abs(y-y_c)<r):
            x = (r - (x_c - x))*ratio
            y = (r - (y_c - y))*ratio
            canvas.DrawCircle(x,y,DendriteTracePoint.ZOOM_RADIUS)

    def zoom(self, parameters, img):
        x,y,z = self.point

        # find the region of interest
        summary = dt_image.roi(parameters,self)
        bot_edge,top_edge,left_edge,right_edge = summary['edges']
        bot_bounds,top_bounds,left_bounds,right_bounds = summary['bounds']

        # take only the layer plus and minus the zoom_depth
        zoom_size = parameters['zoom_size']
        zoom_depth = parameters['zoom_depth']
        size = (zoom_size,zoom_size,zoom_depth*2+1)
        zoom_img = numpy.zeros(size)

        size = img.shape
        for i in range(-zoom_depth,zoom_depth+1):
            index = i + z
            if(index >= 0 and index < size[2]):
                section = img[bot_bounds:top_bounds,
                    left_bounds:right_bounds,index]
                zoom_img[bot_edge:top_edge,left_edge:right_edge,i] = section

        return dt_image.proj_image(zoom_img)

    def dist(self,other):
        x_0,y_0,z_0 = self.point
        x_1,y_1,z_1 = other.point
        return math.sqrt((x_0-x_1)**2 + (y_0-y_1)**2 + (z_0-z_1)**2)

    def __eq__(self,other):
        if(not isinstance(other,DendriteTracePoint)):
            return False
        return self.point == other.point

    def __repr__(self):
        x,y,z = self.point
        return str(x) + ',' + str(y) + ',' + str(z)

    def __hash__(self):
        return hash(self.point)