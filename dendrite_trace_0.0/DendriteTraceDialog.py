
########################## Dendrite Trace Dialog ##########################
# This is a file that contains dialog classes for Dendrite Trace for the 
# user interface.
###########################################################################

import copy
import numpy
import math
import json
import pickle
import os
import wx
from wx import *
import numpy
import scipy.misc
import DendriteTraceImage as dt_image
import DendriteTraceMachineLearning as ml

class SetZoomSizeDialog(wx.Dialog):
    SIZE1 = 128
    SIZE2 = 64
    SIZE3 = 32

    def __init__(self, *args, **kw):
        super(SetZoomSizeDialog, self).__init__(*args, **kw) 
            
        self.init()
        self.size = (250,200)
        self.title = 'Set Zoom Size'
        self.SetSize(self.size)
        self.SetTitle(self.title)
        
        
    def init(self):
        self.main_panel = wx.Panel(self)
        self.main_sizer = wx.BoxSizer(wx.VERTICAL)

        size_box = wx.StaticBox(
            self.main_panel, label='Zoom Box Sizes')
        self.size_box = wx.StaticBoxSizer(size_box, wx.VERTICAL)    
        self.radio_button_1 = wx.RadioButton(
            self.main_panel, label=str(SetZoomSizeDialog.SIZE1) + ' Pixels') 
        self.radio_button_2 = wx.RadioButton(
            self.main_panel, label=str(SetZoomSizeDialog.SIZE2) + ' Pixels')
        self.radio_button_3 = wx.RadioButton(
            self.main_panel, label=str(SetZoomSizeDialog.SIZE3) + ' Pixels')
        self.radio_button_4 = wx.RadioButton(self.main_panel, label='Custom')
        self.custom_box = wx.TextCtrl(self.main_panel)
        
        self.set_size_box = wx.BoxSizer(wx.HORIZONTAL)        
        self.size_box.Add(self.radio_button_1)
        self.size_box.Add(self.radio_button_2)
        self.size_box.Add(self.radio_button_3)
        self.set_size_box.Add(self.radio_button_4)
        self.set_size_box.Add(self.custom_box,flag=wx.LEFT, border=5)
        self.size_box.Add(self.set_size_box)
        
        self.main_panel.SetSizer(self.size_box)
       
        self.done_box = wx.BoxSizer(wx.HORIZONTAL)
        ok_button = wx.Button(self, wx.ID_OK, label='Ok')
        close_button = wx.Button(self, label='Close')
        self.done_box.Add(ok_button)
        self.done_box.Add(close_button, flag=wx.LEFT, border=5)

        self.main_sizer.Add(self.main_panel, proportion=1, 
            flag=wx.ALL|wx.EXPAND, border=5)
        self.main_sizer.Add(self.done_box, 
            flag=wx.ALIGN_CENTER|wx.TOP|wx.BOTTOM, border=10)

        self.SetSizer(self.main_sizer)
        
        close_button.Bind(wx.EVT_BUTTON, self.on_quit)


    def on_quit(self, event):
        self.Destroy()


class ManualAutomaticSelection(wx.Dialog):
    def __init__(self, *args, **kw):
        super(ManualAutomaticSelection, self).__init__(*args, **kw) 
            
        self.init()
        self.size = (250,200)
        self.title = 'Selection'
        self.SetSize(self.size)
        self.SetTitle(self.title)
        
        
    def init(self):
        self.main_panel = wx.Panel(self)
        self.main_sizer = wx.BoxSizer(wx.VERTICAL)

        size_box = wx.StaticBox(
            self.main_panel, label='Tracing Method')
        self.size_box = wx.StaticBoxSizer(size_box, wx.VERTICAL)
        self.manual = wx.RadioButton(self.main_panel, 
            label='Manual Trace (Press M)')
        self.auto = wx.RadioButton(self.main_panel, 
            label='Auto Trace (Press A)')    
        self.size_box.Add(self.manual)
        self.size_box.Add(self.auto)
        
        self.main_panel.SetSizer(self.size_box)
       
        self.done_box = wx.BoxSizer(wx.HORIZONTAL)
        ok_button = wx.Button(self, wx.ID_OK, label='Ok')
        close_button = wx.Button(self, label='Close')
        self.done_box.Add(ok_button)
        self.done_box.Add(close_button, flag=wx.LEFT, border=5)

        self.main_sizer.Add(self.main_panel, proportion=1, 
            flag=wx.ALL|wx.EXPAND, border=5)
        self.main_sizer.Add(self.done_box, 
            flag=wx.ALIGN_CENTER|wx.TOP|wx.BOTTOM, border=10)

        self.SetSizer(self.main_sizer)
        close_button.Bind(wx.EVT_BUTTON, self.on_quit)

        # add key down event handler
        self.manual.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
        self.auto.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
        self.manual.SetFocus()
        
        
    def on_quit(self, event):
        self.Destroy()


    def on_key_down(self, event):
        keycode = event.GetKeyCode()
        if(keycode == ord('M')):
            self.manual.SetValue(True)
            self.manual.SetFocus()
        elif(keycode == ord('A')):
            self.auto.SetValue(True)
            self.auto.SetFocus()
        elif(keycode == wx.WXK_RETURN):
            self.EndModal(wx.ID_OK)
        elif(keycode == wx.WXK_UP or
            keycode == wx.WXK_DOWN):
            if(self.manual.GetValue()):
                self.auto.SetValue(True)
            else:
                self.manual.SetValue(True)


class TrainingData(wx.Dialog):
    def __init__(self,dendrite_trace):
        super(TrainingData, self).__init__(None)
        self.dendrite_trace = dendrite_trace
        self.curr_zoom_images = list()
        title = 'Info'
        self.SetTitle(title)

        self.main_panel = wx.Panel(self,-1)
        self.main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.button_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.count = 0
        # gauge_len is pos + neg + zoom_img, and where len(pos) = len(neg)
        self.gauge_len = (2*len(self.dendrite_trace.theta_list) + 
            len(self.dendrite_trace.zoom_image_list))

        self.gauge = wx.Gauge(self.main_panel,wx.ID_ANY,self.gauge_len,
            size=(250, 25))
        self.text = wx.StaticText(self.main_panel,wx.ID_ANY,
            label='Press generate to generate data')
        self.start_button = wx.Button(
            self.main_panel,wx.ID_ANY,label='Generate')
        self.close_button = wx.Button(
            self.main_panel,wx.ID_ANY,label='Close')

        self.button_sizer.Add(self.start_button,0,0)
        self.button_sizer.AddSpacer(40)
        self.button_sizer.Add(self.close_button,0,0)

        self.main_sizer.AddSpacer(40)
        self.main_sizer.Add(self.gauge,0,wx.ALIGN_CENTER)
        self.main_sizer.AddSpacer(40)
        self.main_sizer.Add(self.button_sizer,0,wx.ALIGN_CENTER)
        self.main_sizer.AddSpacer(40)
        self.main_sizer.Add(self.text,0,wx.ALIGN_CENTER)

        self.start_button.Bind(wx.EVT_BUTTON,self.on_start,self.start_button)
        self.close_button.Bind(wx.EVT_BUTTON,self.on_quit,self.close_button)

        self.main_panel.SetSizer(self.main_sizer)
        self.Centre()


    def load(self):
        database = read_file(self.dendrite_trace.ml_data_path)
        if(database == ''):
            self.database = dict()
            self.zoom_count = 0 # total number of zoom images
            self.zoom_images = list() # all zoom images
            self.positive_count = 0
            self.negative_count = 0
            self.one_count = 0
            self.two_count = 0
            self.three_count = 0
        else:
            self.database = json.loads(database)
            self.zoom_count = self.database['zoom_count']
            self.zoom_images = self.database['zoom_images']
            self.positive_count = self.database['positive_count']
            self.negative_count = self.database['negative_count']
            self.one_count = self.database['one_count']
            self.two_count = self.database['two_count']
            self.three_count = self.database['three_count']


    def save(self):
        self.database['zoom_count'] = self.zoom_count
        self.database['zoom_images'] = self.zoom_images
        self.database['positive_count'] = self.positive_count
        self.database['negative_count'] = self.negative_count
        self.database['one_count'] = self.one_count
        self.database['two_count'] = self.two_count
        self.database['three_count'] = self.three_count
        database = json.dumps(self.database)
        write_file(self.dendrite_trace.ml_data_path, database)


    def create_zoom_image(self):
        # note: zoom_img is a three tuple of the center of the actual zoom_img
        for zoom_img in self.dendrite_trace.zoom_images:
            image_name = 'zoom_image_' + str(self.zoom_count) + '.tif'
            self.save_zoom_image(zoom_img,image_name)
            self.zoom_images += [zoom_img]
            self.zoom_count += 1


    def save_zoom_image(self,zoom_img,name):
        x_c,y_c,z_c = zoom_img
        self.dendrite_trace.x_coord = x_c
        self.dendrite_trace.y_coord = y_c
        self.dendrite_trace.z_coord = z_c
        dt_image.zoom_3d(self.dendrite_trace)
        zoom_img = dt_image.proj_image(self.dendrite_trace.zoom_img)
        img_path = os.path.join(self.dendrite_trace.image_path,name)
        scipy.misc.imsave(img_path,zoom_img)
        self.curr_zoom_images += [zoom_img]


    def create_training_data(self):
        theta_list = self.dendrite_trace.theta_list
        zoom_images = self.zoom_images
        dt = self.dendrite_trace

        reward_path = os.path.join(dt.reward_path,dt.plus_path)
        value_path = os.path.join(dt.value_path,dt.plus_path)

        # positive training data
        for i in range(len(theta_list)):
            img = self.curr_zoom_images[i]
            theta = theta_list[i]
            reward_img = dt_image.reward(dt,img,theta)
            value_img = dt_image.value(dt,img,theta)

            reward_name = 'reward_' + str(self.positive_count) + '.tif'
            value_name = 'value_' + str(self.positive_count) + '.tif'
            reward_name = os.path.join(reward_path,reward_name)
            value_name = os.path.join(value_path,value_name)

            scipy.misc.imsave(reward_name,reward_img)
            scipy.misc.imsave(value_name,value_img)

            self.positive_count += 1
            self.increment_gauge()
            self.Update()

        reward_path = os.path.join(dt.reward_path,dt.minus_path)
        value_path = os.path.join(dt.value_path,dt.minus_path)

        theta_index = 0
        # negative training data
        for i in range(len(dt.num_connections)):
            num_connect = dt.num_connections[i]
            curr_theta_list = dt.theta_list[theta_index:theta_index+num_connect]
            theta_list = list()
            if(num_connect == 1):
                # if one connection, add 180 to the angle
                theta_list = [curr_theta_list[0]+180]
            elif(num_connect == 2):
                # if two connections, take half of each of two arcs
                theta_0,theta_1 = curr_theta_list[0],curr_theta_list[1]
                theta_max,theta_min = max(curr_theta_list),min(curr_theta_list)
                diff = theta_max - theta_min
                theta_list = [theta_min + diff/2,theta_min - (360-diff)/2]
            elif(num_connect == 3):
                # if three connections, take half of each of three arcs
                theta_max,theta_min = max(curr_theta_list),min(curr_theta_list)
                curr_theta_list.remove(theta_min)
                curr_theta_list.remove(theta_max)
                theta_mid = curr_theta_list.pop()
                theta_list += [(theta_mid-theta_min)/2+theta_min]
                theta_list += [(theta_max-theta_mid)/2+theta_mid]
                theta_list += [(theta_min+360-theta_max)/2+theta_max]

            for j in range(len(theta_list)):
                img = self.curr_zoom_images[i]
                theta = theta_list[j]
                reward_img = dt_image.reward(dt,img,theta)
                value_img = dt_image.value(dt,img,theta)

                reward_name = 'reward_' + str(self.negative_count) + '.tif'
                value_name = 'value_' + str(self.negative_count) + '.tif'
                reward_name = os.path.join(reward_path,reward_name)
                value_name = os.path.join(value_path,value_name)

                scipy.misc.imsave(reward_name,reward_img)
                scipy.misc.imsave(value_name,value_img)

                self.negative_count += 1
                self.increment_gauge()

            theta_index += num_connect

        self.increment_gauge()


    def create_connection_data(self):
        dt = self.dendrite_trace
        for i in range(len(dt.zoom_image_list)):
            num = dt.num_connections[i]
            img = dt.zoom_image_list[i]
            dt.x_coord,dt.y_coord,dt.z_cord = img
            dt_image.zoom_3d(dt)
            img = dt_image.proj_image(dt.zoom_img)
            path = ml.CONNECTION_PATH
            path = os.path.join(path,str(num))
            if(num == 1):
                count_path = 'one_' + str(self.one_count) + '.tif'
                self.one_count += 1
            elif(num == 2):
                count_path = 'two_' + str(self.two_count) + '.tif'
                self.two_count += 1
            elif(num == 3):
                count_path = 'three_' + str(self.three_count) + '.tif'
                self.three_count += 1
            path = os.path.join(path,count_path)
            scipy.misc.imsave(path,img)

            self.increment_gauge


    def increment_gauge(self):
        if(self.count < self.gauge_len):
            self.text.SetLabel('Generating training data...')
            self.count += 1
            self.gauge.SetValue(self.count)
            self.Update()
        else:
            self.text.SetLabel('Done generating training data')


    def on_start(self,event):
        self.load()
        self.create_zoom_image()
        self.create_training_data()
        self.create_connection_data()
        self.save()


    def on_quit(self,event):
        self.Destroy()


# from course notes
def read_file(path):
    with open(path, "rt") as f:
        return f.read()

def write_file(path, contents):
    with open(path, "wt") as f:
        f.write(contents)

def almost_equal_num(n1,n2):
    return abs(n1-n2) < 10**-5

# almost_equal for tuples (x,y,z)
def almost_equal(t1,t2):
    return (almost_equal_num(t1[0],t2[0]) and 
            almost_equal_num(t1[1],t2[1]) and 
            almost_equal_num(t1[2],t2[2]))


class TrainML(wx.Dialog):
    BRANCH =   'Branch'
    CONNECTION = 'Connection'
    def __init__(self,mode):
        super(TrainML, self).__init__(None)
        # determines whether training connection or branch
        self.mode = mode
        title = 'Retrain %s ML' %self.mode
        self.SetTitle(title)

        self.main_panel = wx.Panel(self,-1)
        self.main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.button_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.count = 0
        self.gauge_len = 2 # load data and train on data

        self.gauge = wx.Gauge(self.main_panel,wx.ID_ANY,self.gauge_len,
            size=(250, 25))
        self.text = wx.StaticText(self.main_panel,wx.ID_ANY,
            label='Press train to train the algorithm')
        self.start_button = wx.Button(
            self.main_panel,wx.ID_ANY,label='Train')
        self.close_button = wx.Button(
            self.main_panel,wx.ID_ANY,label='Close')

        self.button_sizer.Add(self.start_button,0,0)
        self.button_sizer.AddSpacer(40)
        self.button_sizer.Add(self.close_button,0,0)

        self.main_sizer.AddSpacer(40)
        self.main_sizer.Add(self.gauge,0,wx.ALIGN_CENTER)
        self.main_sizer.AddSpacer(40)
        self.main_sizer.Add(self.button_sizer,0,wx.ALIGN_CENTER)
        self.main_sizer.AddSpacer(40)
        self.main_sizer.Add(self.text,0,wx.ALIGN_CENTER)

        self.start_button.Bind(wx.EVT_BUTTON,self.on_start,self.start_button)
        self.close_button.Bind(wx.EVT_BUTTON,self.on_quit,self.close_button)

        self.main_panel.SetSizer(self.main_sizer)
        self.Centre()


    def on_start(self,event):
        self.train()


    def train(self):
        self.text.SetLabel('Loading training data to train on...')
        self.Update()
        if(self.mode == CONNECTION):
            dataset = ml.load_connection_data()
        elif(self.mode == BRANCH):
            dataset = ml.load_branch_data()
        self.increment_gauge()
        self.text.SetLabel('Training the ML algorithm...')
        self.Update()
        if(self.mode == CONNECTION):
            ml.train_connection_ml(dataset)
        elif(self.mode == BRANCH):
            ml.train_branch_ml(dataset)
        self.increment_gauge()
        self.increment_gauge() # increment again to display done message


    def increment_gauge(self):
        if(self.count < self.gauge_len):
            self.count += 1
            self.gauge.SetValue(self.count)
            self.Update()
        else:
            self.text.SetLabel('Done training the ML algorithm')
            self.Update()


    def on_quit(self,event):
        self.Destroy()


class TrainingDataGeneration(wx.Dialog):

    def __init__(self,dendrite_trace):
        super(TrainingDataGeneration, self).__init__(None)
        self.dendrite_trace = dendrite_trace
        title = 'Info'
        self.SetTitle(title)

        self.main_panel = wx.Panel(self,-1)
        self.main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.button_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.count = 0
        self.gauge_len = len(self.dendrite_trace.dendrite_points)-1

        self.gauge = wx.Gauge(self.main_panel,wx.ID_ANY,self.gauge_len,
            size=(250, 25))
        self.text = wx.StaticText(self.main_panel,wx.ID_ANY,
            label='Press generate to generate data')
        self.start_button = wx.Button(
            self.main_panel,wx.ID_ANY,label='Generate')
        self.close_button = wx.Button(
            self.main_panel,wx.ID_ANY,label='Close')

        self.button_sizer.Add(self.start_button,0,0)
        self.button_sizer.AddSpacer(40)
        self.button_sizer.Add(self.close_button,0,0)

        self.main_sizer.AddSpacer(40)
        self.main_sizer.Add(self.gauge,0,wx.ALIGN_CENTER)
        self.main_sizer.AddSpacer(40)
        self.main_sizer.Add(self.button_sizer,0,wx.ALIGN_CENTER)
        self.main_sizer.AddSpacer(40)
        self.main_sizer.Add(self.text,0,wx.ALIGN_CENTER)

        self.start_button.Bind(wx.EVT_BUTTON,self.on_start,self.start_button)
        self.close_button.Bind(wx.EVT_BUTTON,self.on_quit,self.close_button)

        self.main_panel.SetSizer(self.main_sizer)
        self.Centre()


    def load(self):
        if(read_file(self.dendrite_trace.database_path) == ''):
            self.database = dict()
            self.positive_count = 0
            self.negative_count = 0
            self.positive = dict()
            self.negative = dict()
        else:
            f = open(self.dendrite_trace.database_path,'rb')
            self.database = pickle.load(f)
            self.positive_count = self.database['positive_count']
            self.negative_count = self.database['negative_count']
            self.positive = self.database['positive']
            self.negative = self.database['negative']


    def save(self):
        self.database['positive_count'] = self.positive_count
        self.database['negative_count'] = self.negative_count
        self.database['positive'] = self.positive
        self.database['negative'] = self.negative
        f = open(self.dendrite_trace.database_path,'wb')
        pickle.dump(self.database,f)


    def create_training_data(self):
        dt = self.dendrite_trace

        def on_predict(dt,point,theta_naught):
            # center point
            x_c,y_c,z_c = point.point
            parameters = dt.parameters
            predict_list = list()
            for theta in range(0,360,6):
                theta = (theta + theta_naught)%360
                # zoom in at each point depth
                r,d = parameters['manual_radius'],parameters['zoom_depth']
                x_zoom,y_zoom = r*math.cos(theta),r*math.sin(theta)
                x,y,z = x_zoom+x_c,y_zoom+y_c,z_c
                # indices reversed row,col = y,x
                try: z = numpy.argmax(dt.img[y,x,z-d:z+d])-d+z
                except: z = 1
                point = copy.copy(dt.curr_point)
                point.point = (x_c,y_c,z)
                img = dt_image.zoom(dt,point)
                # extract input images
                prediction = dt.predict(img,theta)
                predict_list.append(prediction)
            return predict_list

        good_connections = list()
        bad_connections = list()
        for point in dt.dendrite_points:
            # positive connections
            for theta in point.theta_list:
                data = list()
                data.append(point)
                data.append(theta)
                data.append(on_predict(dt,point,theta))
                good_connections.append(data)
            # negative connections
            for theta in point.bad_theta_list:
                data = list()
                data.append(point)
                data.append(theta)
                data.append(on_predict(dt,point,theta))
                bad_connections.append(data)

        def save_data(self,connections,folder,counter,predict):
            for data in connections:
                img = data[0].zoom(dt.parameters,dt.img)
                img = dt_image.reward(dt,img,data[1])
                img_path = os.path.join(folder,'img_'+str(counter)+'.tif')
                scipy.misc.imsave(img_path,img)
                predict_list = data[2]
                predict['predict_list_'+str(counter)] = predict_list
                counter += 1
                self.increment_gauge()
            return counter,predict

        self.positive_count,self.positive = save_data(self,good_connections,
            ml.POS,self.positive_count,self.positive)
        self.negative_count,self.negative = save_data(self,bad_connections,
            ml.NEG,self.negative_count,self.negative)

        self.save()

        self.increment_gauge()


    def increment_gauge(self):
        if(self.count < self.gauge_len):
            self.text.SetLabel('Generating training data...')
            self.count += 1
            self.gauge.SetValue(self.count)
            self.Update()
        else:
            self.text.SetLabel('Done generating training data')


    def on_start(self,event):
        self.load()
        self.create_training_data()
        self.save()


    def on_quit(self,event):
        self.Destroy()