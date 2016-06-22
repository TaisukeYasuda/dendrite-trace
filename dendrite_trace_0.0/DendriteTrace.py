
############################## Dendrite Trace 0.0 ############################## 

import pickle
import math
import copy
import os
import wx
import cv2
import numpy
import scipy
import scipy.misc
from sklearn import cross_validation as cv
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import DendriteTraceImage as dt_image
import DendriteTraceDialog as dt_dialog
import DendriteTraceMachineLearning as ml
from DendriteTracePoint import DendriteTracePoint as dt_point
from DendriteTraceDialog import *
from PIL import Image

class DendriteTrace(wx.Frame):
    # path constants
    LINE = '-------------------------------------'
    DENDRITE = 0
    MOVE = 1
    IMAGE_PATH = './data/temp.tif'
    DATA_PATH = './data/'
    VALUE_PATH = ml.VALUE_PATH
    REWARD_PATH = ml.REWARD_PATH
    PLUS_PATH = ml.PLUS_PATH
    MINUS_PATH = ml.MINUS_PATH
    CONNECTION_PATH = ml.CONNECTION_PATH
    IMAGES_PATH = './data/images'
    ML_DATA_PATH = './data/ml_data.txt'
    DATABASE_PATH = './data/database.txt'

    DOCS = '''
    Welcome to Dendrite Trace 0.0! This is a scientific tool 
    for neuroscientists who would like to analyze the structure 
    of neurons from three dimensional fluorescent images. 

    Begin the process by opening an image through the 
    menubar or the button. Then, just follow the instructions 
    on the console through the process. 

    Enjoy!
    '''
    

    def __init__(self):
        super(DendriteTrace, self).__init__(None, -1, 
            style=wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER)
        self.init()
        self.Show()
        self.Centre()

################################ Setting up GUI ################################

    def init(self):
        title = 'Dendrite Trace 0.0'
        size = (1300,600)
        self.SetTitle(title)
        self.SetSize(size)

        self.parameters = dict()

        self.algorithm_variables()
        self.gui_variables()
        self.interface_variables()
        self.path_variables()
        self.add_menubar()
        self.add_layout()
        self.add_event_handlers()


    def algorithm_variables(self):
        self.priority_queue = list()
        self.dendrite_points = list()
        self.initial_point = None

        # numpy array of the images
        self.img = None
        self.zoom_img = None

        self.parameters['value_width'] = 15
        self.parameters['value_height'] = 30
        self.parameters['value_size'] = (15,30)
        self.parameters['reward_width'] = 15
        self.parameters['reward_height'] = 15
        self.parameters['reward_size'] = (15,15)
        self.parameters['zoom_depth'] = 3

        self.dist_limit = 5
        self.cursor = 0

        # load machine learning data
        self.load_ml()

    def load_ml(self):
        # load machine learning models
        f = open(ml.LOG_REG_REWARD_PATH,'rb')
        self.reward_model = pickle.load(f)
        f = open(ml.LOG_REG_VALUE_PATH,'rb')
        self.value_model = pickle.load(f)
        f = open(ml.TREE_REWARD_PATH,'rb')
        self.reward_model_tree = pickle.load(f)
        f = open(ml.TREE_VALUE_PATH,'rb')
        self.value_model_tree = pickle.load(f)
        f = open(ml.REWARD_PCA_PATH,'rb')
        self.reward_pca = pickle.load(f)
        f = open(ml.VALUE_PCA_PATH,'rb')
        self.value_pca = pickle.load(f)

        f = open(ml.PCA_PATH,'rb')
        self.pca = pickle.load(f)
        f = open(ml.LOG_REG_PATH,'rb')
        self.log_reg = pickle.load(f)
        f = open(ml.SVM_PATH,'rb')
        self.svm = pickle.load(f)
        f = open(ml.TREE_PATH,'rb')
        self.tree = pickle.load(f)


    def gui_variables(self):
        self.max_image_size = 512
        self.img_margin = 20
        self.horizontal = 5
        self.control_width = 300
        self.panel_margin = 0
        self.coord_size = 50
        self.terminal_height = 100

        max_image_size = 512
        self.parameters['max_image_size'] = max_image_size


    def interface_variables(self):
        self.dendrite_points_draw = []
        self.clear = False
        self.select_mode = False
        self.manual_mode = False
        self.auto_mode = True
        self.fix_mode = False
        self.theta = 0
        self.theta_velocity = 5
        self.theta_velocity_max = 30
        self.undo_list = list()
        self.redo_list = list()
        self.predict_mode,self.classify_mode = False,False
        self.predict_list,self.classify_list = list(),list()
        self.midpoint_list = list()

        self.prev_point,self.curr_point = None,None


        self.parameters['zoom_size'] = 64
        self.parameters['select_radius'] = 8
        self.parameters['select_velocity'] = 1
        self.parameters['manual_radius'] = 8


    def path_variables(self):
        self.data_path = DendriteTrace.DATA_PATH
        self.value_path = DendriteTrace.VALUE_PATH
        self.reward_path = DendriteTrace.REWARD_PATH
        self.plus_path = DendriteTrace.PLUS_PATH
        self.minus_path = DendriteTrace.MINUS_PATH
        self.connection_path = DendriteTrace.CONNECTION_PATH
        self.image_path = DendriteTrace.IMAGES_PATH
        self.temp_image_path = DendriteTrace.IMAGE_PATH
        self.ml_data_path = DendriteTrace.ML_DATA_PATH
        self.database_path = DendriteTrace.DATABASE_PATH


    def add_layout(self):
        self.main_panel = wx.Panel(self)

        self.main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.image_panel = None
        self.control_panel = None

        # layout that divides the screen horizontally into image and control
        self.add_image_layout()
        self.add_control_layout()

        self.main_sizer.Add(
            self.image_panel,0,wx.EXPAND|wx.ALL,self.panel_margin)
        self.main_sizer.Add(
            self.control_panel,0,wx.EXPAND|wx.ALL,self.panel_margin)

        self.main_panel.SetSizer(self.main_sizer)
        self.main_sizer.Fit(self)
        self.main_panel.Layout()


    def add_event_handlers(self):
        self.image_panel.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
        self.terminal.Bind(wx.EVT_KEY_DOWN, self.on_key_down)


    def add_menubar(self):
        self.menubar = wx.MenuBar()
        self.file_menu,self.edit_menu = wx.Menu(),wx.Menu()
        self.help_menu,self.settings_menu = wx.Menu(),wx.Menu()

        self.open_item = self.file_menu.Append(
            wx.ID_OPEN,'&Open\tCtrl+O','Open image')
        self.quit_item = self.file_menu.Append(
            wx.ID_EXIT,'&Quit\tCtrl+Q','Quit application')

        self.undo_item = self.edit_menu.Append(
            wx.ID_UNDO,'&Undo\tCtrl+Z','Undo point')
        self.redo_item = self.edit_menu.Append(
            wx.ID_REDO,'&Redo\tCtrl+Y','Redo point')

        self.zoom_size_item = self.settings_menu.Append(wx.ID_ANY, 
            'Set Zoom Size (Not recommended)', 'Set zoom size')
        self.clear_terminal_item = self.settings_menu.Append(
            wx.ID_ANY, 'Clear Console')
        self.train_connection_item = self.settings_menu.Append(
            wx.ID_ANY, 'Retrain Connection ML')
        self.train_branch_item = self.settings_menu.Append(
            wx.ID_ANY, 'Retrain Branch ML')

        self.doc_item = self.help_menu.Append(wx.ID_ANY,'Documentation',
            'Documentation')

        self.link_menubar()


    def link_menubar(self):
        self.menubar.Append(self.file_menu, '&File')
        self.menubar.Append(self.edit_menu, '&Edit')
        self.menubar.Append(self.settings_menu, '&Settings')
        self.menubar.Append(self.help_menu, '&Help')

        self.add_menu_events()
        self.SetMenuBar(self.menubar)


    def add_menu_events(self):
        self.Bind(wx.EVT_MENU, self.on_open, self.open_item)
        self.Bind(wx.EVT_MENU, self.on_quit, self.quit_item)
        self.Bind(wx.EVT_MENU, self.on_undo, self.undo_item)
        self.Bind(wx.EVT_MENU, self.on_redo, self.redo_item)
        self.Bind(
            wx.EVT_MENU, self.on_zoom_size, self.zoom_size_item)
        self.Bind(
            wx.EVT_MENU, self.on_clear_terminal, self.clear_terminal_item)
        self.Bind(
            wx.EVT_MENU, self.on_train_connection, self.train_connection_item)
        self.Bind(
            wx.EVT_MENU, self.on_train_branch, self.train_branch_item)
        self.Bind(
            wx.EVT_MENU, self.on_docs, self.doc_item)


    def on_docs(self,event):
        wx.MessageBox(DendriteTrace.DOCS, 'Info', wx.OK | wx.ICON_INFORMATION)


    def add_image_layout(self):
        self.image_panel = wx.Panel(self.main_panel)
        self.image_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.temp_img = wx.EmptyImage(self.max_image_size,self.max_image_size)

        self.add_image()
        self.add_zoom_image()

        self.image_panel.SetSizer(self.image_sizer)
        self.image_panel.Layout()


    def add_image(self):
        # create a display layout for the image
        self.img_box = wx.StaticBox(self.image_panel,wx.ID_ANY,'Image')
        self.img_sizer = wx.StaticBoxSizer(self.img_box,wx.VERTICAL)

        self.img_disp = wx.StaticBitmap(
            self.image_panel, wx.ID_ANY, wx.BitmapFromImage(self.temp_img))
        self.img_sizer.Add(self.img_disp,0,wx.ALL,self.img_margin)

        self.image_sizer.Add(self.img_sizer,0,wx.ALL,
            self.horizontal)


    def add_zoom_image(self):
        # create a display layout for the zoom image
        self.zoom_img_box = wx.StaticBox(
            self.image_panel,wx.ID_ANY,'Zoom Image')
        self.zoom_img_sizer = wx.StaticBoxSizer(self.zoom_img_box,wx.VERTICAL)

        self.zoom_img_disp = wx.StaticBitmap(
            self.image_panel, wx.ID_ANY, wx.BitmapFromImage(self.temp_img))
        self.zoom_img_sizer.Add(self.zoom_img_disp,0,wx.ALL,self.img_margin)

        self.image_sizer.Add(self.zoom_img_sizer,0,wx.ALL,
            self.horizontal)


    def add_control_layout(self):
        self.control_panel = wx.Panel(self.main_panel)
        self.control_sizer = wx.BoxSizer(wx.VERTICAL)

        # adding controls
        self.add_terminal_control()
        self.add_general_control()
        self.add_manual_control()
        self.add_automatic_control()

        self.control_panel.SetSizer(self.control_sizer)
        self.control_panel.Layout()


    def add_general_control(self):
        self.general_box = wx.StaticBox(self.control_panel,wx.ID_ANY,
            label='Start')
        self.general_sizer = wx.StaticBoxSizer(self.general_box,wx.VERTICAL)

        # button to open image
        self.open_button = wx.Button(self.control_panel,wx.ID_ANY,
            label='Open Image')
        self.general_sizer.Add(self.open_button,0,0|wx.CENTER,
            self.horizontal)
        self.Bind(wx.EVT_BUTTON, self.on_open, self.open_button)

        self.show_image_path()
        self.show_selected_points()
        self.show_select_velocity()

        self.control_sizer.Add(self.general_sizer,0,wx.EXPAND,
            self.horizontal)


    def show_image_path(self):
        # text control to show image path folder
        img_path_folder = wx.StaticText(
            self.control_panel,wx.ID_ANY,label='Image Folder:')
        self.general_sizer.Add(
            img_path_folder,0,wx.EXPAND,self.horizontal)
        self.img_path_folder = wx.TextCtrl(self.control_panel,
            style=wx.TE_READONLY,size=(self.control_width,-1))
        self.general_sizer.Add(
            self.img_path_folder,0,wx.EXPAND,self.horizontal)

        # text control to show image path file
        img_path_file = wx.StaticText(
            self.control_panel,wx.ID_ANY,label='Image File:')
        self.general_sizer.Add(
            img_path_file,0,wx.EXPAND,self.horizontal)
        self.img_path_file = wx.TextCtrl(self.control_panel,
            style=wx.TE_READONLY,size=(self.control_width,-1))
        self.general_sizer.Add(
            self.img_path_file,0,wx.EXPAND,self.horizontal)


    def show_selected_points(self):
        # text control to show points selected
        self.general_sizer.AddSpacer(10)
        self.point_sizer = wx.BoxSizer(wx.HORIZONTAL)

        x_point = wx.StaticText(
            self.control_panel,wx.ID_ANY,label='x:')
        self.point_sizer.Add(x_point,0,wx.EXPAND,self.horizontal)
        self.x_point = wx.TextCtrl(
            self.control_panel,style=wx.TE_READONLY,size=(self.coord_size,-1))
        self.point_sizer.Add(self.x_point,0,wx.EXPAND,self.horizontal)

        self.point_sizer.AddSpacer(10)

        y_point = wx.StaticText(
            self.control_panel,wx.ID_ANY,label='y:')
        self.point_sizer.Add(y_point,0,wx.EXPAND,self.horizontal)
        self.y_point = wx.TextCtrl(
            self.control_panel,style=wx.TE_READONLY,size=(self.coord_size,-1))
        self.point_sizer.Add(self.y_point,0,wx.EXPAND,self.horizontal)

        self.general_sizer.Add(
            self.point_sizer,0,0|wx.CENTER,self.horizontal)


    def show_select_velocity(self):
        # radio button to select moving speed
        self.general_sizer.AddSpacer(10)

        select_velocity = wx.StaticText(
            self.control_panel,wx.ID_ANY,label='Selection Speed:')
        self.general_sizer.Add(select_velocity,0,0)
        self.select_velocity_button_1 = wx.RadioButton(
            self.control_panel,wx.ID_ANY,label='Slow')
        self.select_velocity_button_2 = wx.RadioButton(
            self.control_panel,wx.ID_ANY,label='Medium')
        self.select_velocity_button_3 = wx.RadioButton(
            self.control_panel,wx.ID_ANY,label='Fast')
        self.general_sizer.Add(self.select_velocity_button_1,0,0,
            self.horizontal)
        self.general_sizer.Add(self.select_velocity_button_2,0,0,
            self.horizontal)
        self.general_sizer.Add(self.select_velocity_button_3,0,0,
            self.horizontal)

        self.add_button_events()


    def add_button_events(self):
        self.Bind(wx.EVT_RADIOBUTTON, self.on_select_velocity, 
            self.select_velocity_button_1)
        self.Bind(wx.EVT_RADIOBUTTON, self.on_select_velocity, 
            self.select_velocity_button_2)
        self.Bind(wx.EVT_RADIOBUTTON, self.on_select_velocity, 
            self.select_velocity_button_3)


    def on_zoom_size(self, event):
        size = dt_dialog.SetZoomSizeDialog(None)
        if(size.ShowModal() == wx.ID_OK):
            if(size.radio_button_1.GetValue()):
                self.zoom_size = SetZoomSizeDialog.SIZE1
            elif(size.radio_button_2.GetValue()):
                self.zoom_size = SetZoomSizeDialog.SIZE2
            elif(size.radio_button_3.GetValue()):
                self.zoom_size = SetZoomSizeDialog.SIZE3
            elif(size.radio_button_4.GetValue()):
                try: custom_size = int(size.custom_box.GetValue())
                except: self.show_error('Zoom size must be an integer')
                min_size,max_size = 10,self.max_image_size/2
                if(custom_size < min_size):
                    self.show_error('Zoom size is too small')
                elif(custom_size > max_size):
                    self.show_error('Zoom size is too large')
                else: self.zoom_size = custom_size
            self.redraw_all()

        size.Destroy()


    def on_train_connection(self, event):
        train_ml = TrainML('Connection')
        train_ml.ShowModal()
        self.load_ml()


    def on_train_branch(self, event):
        train_ml = TrainML('Branch')
        train_ml.ShowModal()
        self.load_ml()


    def add_manual_control(self):
        self.manual_box = wx.StaticBox(self.control_panel,wx.ID_ANY,
            label='Manual Trace')
        self.manual_sizer = wx.StaticBoxSizer(self.manual_box,wx.VERTICAL)

        self.button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.manual_sizer.Add(self.button_sizer,0,wx.CENTER,
            self.horizontal)

        # undo and redo buttons
        self.undo_button = wx.Button(self.control_panel,wx.ID_ANY,label='Undo')
        self.button_sizer.Add(self.undo_button,0,wx.EXPAND,
            self.horizontal)
        self.button_sizer.AddSpacer(10)
        self.redo_button = wx.Button(self.control_panel,wx.ID_ANY,label='Redo')
        self.button_sizer.Add(self.redo_button,0,wx.EXPAND,
            self.horizontal)
        self.undo_button.Bind(wx.EVT_BUTTON,self.on_undo,self.undo_button)
        self.redo_button.Bind(wx.EVT_BUTTON,self.on_redo,self.redo_button)

        # predictor buttons
        self.connection = wx.Button(self.control_panel,wx.ID_ANY,
            label='Connection Predictor')
        self.branch = wx.Button(self.control_panel,wx.ID_ANY,
            label='Branch Predictor')
        self.manual_sizer.Add(self.connection,0,wx.CENTER,self.horizontal)
        self.manual_sizer.Add(self.branch,0,wx.CENTER,self.horizontal)
        self.connection.Bind(wx.EVT_BUTTON,self.show_connection,self.connection)
        self.branch.Bind(wx.EVT_BUTTON,self.show_branch,self.branch)

        self.control_sizer.Add(self.manual_sizer,0,wx.EXPAND,
            self.horizontal)


    def show_connection(self,event):
        self.predict_mode = not self.predict_mode
        self.image_panel.SetFocus()
        self.redraw_all()


    def show_branch(self,event):
        self.classify_mode = not self.classify_mode
        self.image_panel.SetFocus()
        self.redraw_all()


    def on_predict(self):
        x_c,y_c,z_c = self.curr_point.point
        if(z_c != None):
            parameters = self.parameters
            self.predict_list = list()
            for theta in range(0,360,6):
                # zoom in at each point depth
                r,d = parameters['manual_radius'],parameters['zoom_depth']
                x_zoom,y_zoom = r*math.cos(theta),r*math.sin(theta)
                x,y,z = x_zoom+x_c,y_zoom+y_c,z_c
                # indices reversed row,col = y,x
                try: z = numpy.argmax(self.img[y,x,z-d:z+d])-d+z
                except: z = 1
                point = copy.copy(self.curr_point)
                point.point = (x_c,y_c,z)
                img = dt_image.zoom(self,point)
                # extract input images
                prediction = self.predict(img,theta)
                self.predict_list += [prediction]

            self.redraw_all()

        self.image_panel.SetFocus()


    def predict(self,img,theta):
        reward_img = dt_image.reward(self,img,theta)
        value_img = dt_image.value(self,img,theta)
        # flatten to one dimension
        reward_img = [numpy.ndarray.flatten(reward_img)]
        value_img = [numpy.ndarray.flatten(value_img)]
        # reduce dimension further
        reward_img = self.reward_pca.transform(reward_img)
        value_img = self.value_pca.transform(value_img)
        # get results
        reward_prediction = self.reward_model.predict(reward_img)
        value_prediction = self.value_model.predict(value_img)
        return reward_prediction or value_prediction


    def on_classify(self):
        x_c,y_c,z_c = self.curr_point.point
        if(z_c != None):
            parameters = self.parameters
            self.classify_list = list()
            zoom_img = self.curr_point.zoom(self.parameters,self.img)
            r,d = parameters['manual_radius'],parameters['zoom_depth']
            self.on_predict()
            predict_list = self.predict_list
            for theta in range(0,360,10):
                classification = 0
                img = copy.deepcopy(zoom_img)
                data = dt_image.training_data(self,img,theta).flatten()
                data = numpy.append(data,numpy.average(data))
                data = numpy.append(data,predict_list)
                img = self.pca.transform([data])
                # extract input images2
                prediction = self.log_reg.predict(img)
                self.classify_list.append(prediction)

            self.redraw_all()

        self.image_panel.SetFocus()


    def on_undo(self, event):
        self.image_panel.SetFocus()
        if(len(self.undo_list) > 0 and len(self.dendrite_points)>1):
            undo = self.undo_list.pop()
            if(undo[0] == DendriteTrace.DENDRITE):
                self.curr_point.remove_connection()
                self.priority_queue.pop()
                x,y,z = self.dendrite_points.pop().point
                x,y,z = int(x),int(y),int(z)
                self.log('\t--> Undo point ' + str((x,y,z)))
            elif(undo[0] == DendriteTrace.MOVE):
                self.curr_point = undo[1]
                next_point = undo[2]
                next_point.remove_connection()
                self.priority_queue.append(next_point)
                self.log('\t--> Undo move')
            self.redo_list += [undo]
            self.on_predict()
            self.on_classify()
            self.redraw_all()


    def on_redo(self, event):
        self.image_panel.SetFocus()
        if(len(self.redo_list) > 0):
            redo = self.redo_list.pop()
            if(redo[0] == DendriteTrace.DENDRITE):
                self.curr_point.add_connection(redo[1])
                self.dendrite_points.append(redo[1])
                self.priority_queue.append(redo[1])
                x,y,z = redo[1].point
                x,y,z = int(x),int(y),int(z)
                self.log('\t--> Redo point ' + str((x,y,z)))
            elif(redo[0] == DendriteTrace.MOVE):
                self.log('\t--> Redo move')
                self.prev_point = self.curr_point
                self.curr_point = self.priority_queue.pop()
                self.curr_point.add_connection(self.prev_point)
            self.undo_list += [redo]
            self.on_predict()
            self.on_classify()
            self.redraw_all()


    def add_automatic_control(self):
        self.auto_box = wx.StaticBox(self.control_panel,wx.ID_ANY,
            label='Auto Trace')
        self.auto_sizer = wx.StaticBoxSizer(self.auto_box,wx.VERTICAL)

        self.trace_button = wx.Button(self.control_panel,
            wx.ID_ANY,label='Trace')
        self.auto_sizer.Add(self.trace_button,0,wx.CENTER,self.horizontal)
        self.trace_button.Bind(wx.EVT_BUTTON,self.auto_trace,self.trace_button)

        self.fix_button = wx.Button(self.control_panel,
            wx.ID_ANY,label='Fix')
        self.auto_sizer.Add(self.fix_button,0,wx.CENTER,self.horizontal)
        self.fix_button.Bind(wx.EVT_BUTTON,self.auto_fix,self.fix_button)


        # add button
        self.add_button = wx.Button(self.control_panel,wx.ID_ANY,label='Add')
        self.add_button.Bind(wx.EVT_BUTTON,self.on_add,self.add_button)

        self.auto_sizer.Add(self.add_button,0,wx.CENTER,self.horizontal)

        self.control_sizer.Add(self.auto_sizer,0,wx.EXPAND,self.horizontal)


    def add_terminal_control(self):
        self.terminal_box = wx.StaticBox(
            self.control_panel,wx.ID_ANY,'Console')
        self.terminal_sizer = wx.StaticBoxSizer(self.terminal_box,wx.VERTICAL)

        self.terminal = wx.TextCtrl(self.control_panel,
            style=wx.TE_READONLY|wx.TE_MULTILINE|wx.TE_WORDWRAP,
            size=(self.control_width,self.terminal_height))
        self.terminal_text = list()
        self.log('Welcome to Dendrite Trace 0.0!')
        self.log('')
        self.log('-Open an image to start')
        self.terminal_sizer.Add(self.terminal,0,wx.EXPAND,
            self.horizontal)

        self.control_sizer.Add(self.terminal_sizer,0,wx.EXPAND,
            self.horizontal)


    def display_terminal(self):
        self.terminal.SetValue('\n'.join(self.terminal_text))
        self.terminal.ShowPosition(self.terminal.GetLastPosition())


    def log(self,text):
        self.terminal_text += [text]
        self.display_terminal()


    def on_quit(self, event):
        quit = wx.MessageBox('Are you sure you want to quit?', 'Quit', 
                wx.YES_NO|wx.NO_DEFAULT, self)
        if(quit == wx.YES):
            self.Close()


    def on_open(self, event):
        parameters = self.parameters
        self.reset()

        open_file_dialog = wx.FileDialog(self, 'Open TIF file', '', '', 
            'TIF files (*.tif)|*.tif',wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        # show the open file dialog
        if(open_file_dialog.ShowModal() == wx.ID_CANCEL):
            self.image_panel.SetFocus()
            return

        # create current point
        mid = parameters['max_image_size']/2
        self.curr_point = dt_point((mid,mid,0))

        # load the image
        self.img_path = open_file_dialog.GetPath()
        img = dt_image.process_image(dt_image.load_image(self.img_path))

        self.img = dt_point.IMG = img # store the numpy image

        self.proj_img = dt_image.proj_image(img)
        self.img_bitmap = dt_image.numpy_to_wxbitmap(self.proj_img)
        self.zoom_bitmap = wx.BitmapFromImage(self.temp_img)

        self.display_open_image()


    def display_open_image(self):
        # display the image and image path on the static bitmap and text control
        self.img_disp.SetBitmap(self.img_bitmap)
        self.zoom_img_disp.SetBitmap(self.zoom_bitmap)
        img_path_file = self.img_path.split('/')[-1]
        img_path_folder = self.img_path[:-len(img_path_file)]
        self.img_path_file.SetValue(img_path_file)
        self.img_path_folder.SetValue(img_path_folder)

        self.log('\t--> New image opened')
        self.log(DendriteTrace.LINE)
        self.log('-Use arrow keys to select initial point')
        self.log('-Press enter when done')
        self.display_terminal()

        # zoom image processing
        self.zoom_img = dt_image.zoom_2d(self)

        self.select_mode = True
        self.image_panel.SetFocus()
        self.redraw_all()


    def on_clear_terminal(self, event):
        self.terminal_text = list()
        self.display_terminal()


    def on_select_velocity(self, event):
        if(self.select_velocity_button_1.GetValue()):
            self.parameters['select_velocity'] = 1
        elif(self.select_velocity_button_2.GetValue()):
            self.parameters['select_velocity'] = 5
        elif(self.select_velocity_button_3.GetValue()):
            self.parameters['select_velocity'] = 20
        self.image_panel.SetFocus()


    def show_error(self, msg):
        wx.MessageBox(msg,'Error',wx.OK)


    def on_key_down(self, event):
        self.image_panel.SetFocus()
        keycode = event.GetKeyCode()
        if(self.select_mode):
            self.key_down_select(keycode)
        elif(self.manual_mode):
            self.key_down_manual(keycode)
        elif(self.auto_mode):
            self.key_down_auto(keycode)


    def key_down_select(self,keycode):
        parameters = self.parameters
        velocity = parameters['select_velocity']
        if(keycode == wx.WXK_ESCAPE): 
            self.clear = not self.clear
        elif(keycode == wx.WXK_UP): 
            self.move_zoom_box(-1*velocity,0)
        elif(keycode == wx.WXK_DOWN): 
            self.move_zoom_box(1*velocity,0)
        elif(keycode == wx.WXK_LEFT): 
            self.move_zoom_box(0,-1*velocity)
        elif(keycode == wx.WXK_RIGHT): 
            self.move_zoom_box(0,1*velocity)
        elif(keycode == wx.WXK_RETURN):
            self.log('\t--> Initial point selected')
            self.log(DendriteTrace.LINE)
            self.log('-Choose a tracing method')
            self.on_done_select_key()
        self.redraw_all()


    def key_down_manual(self,keycode):
        velocity = self.parameters['select_velocity']
        if(keycode == wx.WXK_LEFT):
            self.move_theta(1)
        elif(keycode == wx.WXK_RIGHT): 
            self.move_theta(-1)
        elif(keycode == wx.WXK_UP):
            if(self.theta_velocity < self.theta_velocity_max): 
                self.theta_velocity += 1
        elif(keycode == wx.WXK_DOWN):
            if(self.theta_velocity > 1): 
                self.theta_velocity -= 1
        elif(keycode == wx.WXK_SPACE):
            self.key_down_space()
        elif(keycode == wx.WXK_RETURN):
            self.key_down_return()
        elif(keycode == ord('S')):
            self.predict_mode = not self.predict_mode
            self.redraw_all()
        elif(keycode == ord('C')):
            self.classify_mode = not self.classify_mode
            self.redraw_all()


    def key_down_space(self):
        parameters = self.parameters
        r = parameters['manual_radius']
        x_zoom,y_zoom = r*math.cos(self.theta),r*math.sin(self.theta)
        x,y,z = self.curr_point.point
        x,y,z,d = x_zoom+x,y_zoom+y,z,parameters['zoom_depth']
        # indices reversed row,col = y,x
        try:
            z = numpy.argmax(self.img[y,x,z-d:z+d+1])-d+z
        except:
            height = self.img.shape[2]
            upper_lim = min(z+d,height)
            lower_lim = max(z-d,0)
            diff = z-lower_lim
            z = (numpy.argmax(self.img[y,x,lower_lim:upper_lim+1]) 
                - diff + z)
        if(self.add_dendrite_points(x,y,z)):
            x,y,z = int(x),int(y),int(z)
            self.log('\t--> Point added: ' + str((x,y,z)))
            if(self.fix_mode):
                self.log('-Press enter to move to the added point OR')
                self.log('-Press trace to auto trace OR')
                self.log('-Press fix to move to another place')


    def key_down_return(self):
        self.log('\t--> Next point')
        self.prev_point = self.curr_point
        self.curr_point = self.priority_queue.pop()
        self.curr_point.add_connection(self.prev_point)
        self.on_predict()
        self.on_classify()
        if(len(self.priority_queue) == 0):
            self.log(DendriteTrace.LINE)
            self.log('Done tracing the dendrite!')
            wx.MessageBox('Done tracing the dendrite!',
                'Info',wx.OK)
            self.theta_list = list()
            self.zoom_image_list = list()
            for point in self.dendrite_points:
                self.theta_list += [point.theta_list]
                self.zoom_image_list += [point.point]
                point.calc_bad_theta()
            dt_dialog.TrainingDataGeneration(self).ShowModal()
            #dt_dialog.TrainingData(self).ShowModal()
        # update redo list and undo list
        self.redo_list = list()
        self.undo_list.append((DendriteTrace.MOVE,
            self.prev_point,self.curr_point))
        self.redraw_all()


    def key_down_auto(self,keycode):
        if(keycode == wx.WXK_RETURN):
            self.auto_trace()
        if(self.fix_mode):
            if(keycode == wx.WXK_DOWN):
                if(self.cursor > 1):
                    self.cursor -= 1
                    self.curr_point = self.dendrite_points[self.cursor]
                    self.redraw_all()
            elif(keycode == wx.WXK_UP):
                if(self.cursor < len(self.dendrite_points) - 1):
                    self.cursor += 1
                    self.curr_point = self.dendrite_points[self.cursor]
                    self.redraw_all()


    def move_zoom_box(self, drow, dcol):
        x,y,z = self.curr_point.point
        x += dcol
        y += drow
        max_image_size = self.parameters['max_image_size']
        # check for bounds
        if(x < 0): x = 0
        elif(x > max_image_size): x = max_image_size
        if(y < 0): y = 0
        elif(y > max_image_size): y = max_image_size
        self.curr_point.point = (x,y,z)


    def redraw_all(self):
        self.display_image()
        canvas = self.display_zoom_image()
        self.display_tools(canvas)

        # if appropriate, draw predictions
        if(self.predict_mode):
            self.draw_predict(canvas)
        if(self.classify_mode):
            self.draw_classify(canvas)

        self.zoom_img_disp.SetBitmap(self.zoom_bitmap)
        self.display_coords()
        self.Update()


    def display_image(self):
        x,y,z = self.curr_point.point
        zs = self.parameters['zoom_size']
        # draw dendrite points and zoom box on the original image
        self.img_bitmap = dt_image.numpy_to_wxbitmap(self.proj_img)
        canvas = wx.MemoryDC(self.img_bitmap)
        canvas.SelectObject(self.img_bitmap)
        if(not self.clear):
            canvas.SetBrush(wx.Brush(wx.GREEN))
            canvas.SetPen(wx.TRANSPARENT_PEN)
            draw_list = map(lambda x: x.draw_point,self.dendrite_points)
            canvas.DrawEllipseList(draw_list)
            if(self.select_mode):
                r,color = self.parameters['select_radius'],wx.RED
            else:
                r,color= self.parameters['manual_radius'],wx.BLUE
            canvas.SetPen(wx.Pen(color, 1, wx.SOLID))
            canvas.SetBrush(wx.Brush(wx.CYAN, wx.TRANSPARENT))
            canvas.DrawRectangle(x-zs/2,y-zs/2,zs,zs)
            canvas.DrawCircle(x,y,r)
        self.img_disp.SetBitmap(self.img_bitmap)


    def display_zoom_image(self):
        parameters = self.parameters
        # display the zoom image
        if(self.select_mode):
            self.zoom_bitmap = dt_image.zoom_2d(self)
        else:
            self.zoom_bitmap = dt_image.zoom_3d(self)
        self.zoom_img_disp.SetBitmap(self.zoom_bitmap)
        canvas = wx.MemoryDC(self.zoom_bitmap)
        canvas.SelectObject(self.zoom_bitmap)
        canvas.SetBrush(wx.Brush(wx.GREEN))
        canvas.SetPen(wx.TRANSPARENT_PEN)

        # draw dendrite points on zoom image
        for point in self.dendrite_points:
            point.draw(self.curr_point,parameters,canvas)
        return canvas


    def display_tools(self,canvas):
        parameters = self.parameters
        zoom_size = parameters['zoom_size']
        max_image_size = parameters['max_image_size']
        canvas.SetBrush(wx.Brush(wx.CYAN, wx.TRANSPARENT))
        if(self.select_mode):   
            canvas.SetPen(wx.Pen(wx.RED, 1, wx.SOLID))
            r = parameters['select_radius']
        else:
            canvas.SetPen(wx.Pen(wx.BLUE, 1, wx.SOLID))
            r = parameters['manual_radius']
        # cross
        ratio,mid = max_image_size/zoom_size,max_image_size/2
        canvas.DrawLine(mid,mid-r,mid,mid+r)
        canvas.DrawLine(mid-r,mid,mid+r,mid)
        # circle
        canvas.DrawCircle(mid,mid,r*ratio)
        # display point if in manual mode
        if(self.manual_mode):
            canvas.SetPen(wx.Pen(wx.BLUE, 1, wx.SOLID))
            r = parameters['manual_radius']*ratio
            canvas.DrawCircle(mid+r*math.cos(self.theta),
                mid+r*math.sin(self.theta),dt_point.ZOOM_RADIUS)


    def display_coords(self):
        x,y,z = self.curr_point.point
        self.x_point.SetValue(str(x))
        self.y_point.SetValue(str(y))



    def draw_predict(self,canvas):
        length = len(self.predict_list)
        parameters = self.parameters
        if(length > 0):
            max_image_size = parameters['max_image_size']
            zoom_size = parameters['zoom_size']
            ratio,mid= max_image_size/zoom_size,max_image_size/2
            step = 360/length
            for i in range(length):
                theta = step*i
                prediction = self.predict_list[i]
                if(prediction):
                    canvas.SetPen(wx.Pen(wx.GREEN, 1, wx.SOLID))
                    canvas.SetBrush(wx.Brush(wx.GREEN))
                else:
                    canvas.SetPen(wx.Pen(wx.RED, 1, wx.SOLID))
                    canvas.SetBrush(wx.Brush(wx.RED))
                coeff = 0.5
                r = parameters['manual_radius']*ratio*coeff
                canvas.DrawEllipticArc(mid-r,mid-r,2*r,2*r,
                    theta-step/2,theta+step/2)


    def draw_classify(self,canvas):
        length = len(self.classify_list)
        parameters = self.parameters
        if(length > 0):
            max_image_size = parameters['max_image_size']
            zoom_size = parameters['zoom_size']
            ratio,mid= max_image_size/zoom_size,max_image_size/2
            step = 360/length
            for i in range(length):
                theta = step*i
                prediction = self.classify_list[i]
                if(prediction):
                    canvas.SetPen(wx.Pen(wx.GREEN, 1, wx.SOLID))
                    canvas.SetBrush(wx.Brush(wx.GREEN))
                else:
                    canvas.SetPen(wx.Pen(wx.RED, 1, wx.SOLID))
                    canvas.SetBrush(wx.Brush(wx.RED))
                coeff = 0.5
                r = parameters['manual_radius']*ratio*coeff
                canvas.DrawEllipticArc(mid-r,mid-r,2*r,2*r,
                    theta-step/2,theta+step/2)


    def on_done_select_key(self):
        x,y,z = self.curr_point.point
        # indices reversed row,col = y,x
        z = numpy.argmax(self.img[y,x,:])
        self.initial_point = copy.copy(self.curr_point)
        self.curr_point.point = (x,y,z)
        dt_image.zoom_3d(self)
        self.add_dendrite_points(x,y,z)
        self.on_predict()
        self.on_classify()
        
        selection = dt_dialog.ManualAutomaticSelection(None)
        if(selection.ShowModal() == wx.ID_OK):
            if(selection.manual.GetValue()):
                selection.Destroy()
                self.manual()
            else:
                selection.Destroy()
                self.auto()


    def add_dendrite_points(self,x,y,z):
        r = dt_point.RADIUS
        if(dt_point((x,y,z)) not in self.dendrite_points):
            point,center = (x,y,z),self.curr_point.point
            new_point = dt_point(point,center)
            self.dendrite_points.append(new_point)
            self.priority_queue.append(new_point)
            if(self.select_mode):
                self.curr_point = new_point
            elif(not self.select_mode):
                self.curr_point.add_connection(new_point)
            self.redo_list = list()
            self.undo_list.append((DendriteTrace.DENDRITE,new_point))
            self.redraw_all()
            return True
        return False


    def reset(self):
        self.algorithm_variables()
        self.interface_variables()


    def manual(self):
        self.log('\t--> Manual Trace selected')
        self.log(DendriteTrace.LINE)
        self.manual_mode = True
        self.select_mode = False

        self.log('-Use left and right to move around circle')
        self.log('-Use up and down to change the speed')
        self.log('-Press space to enter points')
        self.log('-Press enter to move to the next point')


    def move_theta(self,sign):
        self.theta += (sign*self.theta_velocity*math.pi/180)%(2*math.pi)
        self.redraw_all()


    def auto(self):
        self.log('\t--> Auto Trace selected')
        self.log(DendriteTrace.LINE)
        self.select_mode = False
        self.auto_mode = True
        self.classify_mode = True
        self.log('-Press the trace button or enter to trace')


    def auto_trace(self,event=None):
        self.log('-Auto trace in progress...')
        self.Update()
        if(self.select_mode or self.manual_mode and not self.fix_mode):
            temp_point = self.priority_queue[0]
            temp_img = self.img
            self.reset()
            self.curr_point = temp_point
            self.img = temp_img
            self.priority_queue.append(self.curr_point)
            self.dendrite_points.append(self.curr_point)
            self.redraw_all()
            self.auto()
        self.manual_mode = False
        while(len(self.priority_queue)>0):
            self.auto_add_point()
            self.auto_move()


    def auto_add_point(self):
        self.on_classify()
        self.on_predict()
        classify_list = self.classify_list
        predict_list = self.predict_list
        midpoint_list_1 = dt_image.find_midpoints(classify_list)
        midpoint_list_2 = dt_image.find_midpoints(predict_list)
        length_1 = len(midpoint_list_1)
        length_2 = len(midpoint_list_2)
        if(length_1 == 2 and length_2 == 2):
            # on_predict is better for non branching segments
            self.midpoint_list = midpoint_list_2
            length = length_2
        else:
            self.midpoint_list = midpoint_list_1
            length = length_1
        parameters = self.parameters
        r,d = parameters['manual_radius'],parameters['zoom_depth']
        max_image_size = parameters['max_image_size']
        point_list = list()
        if(length > 0):
            # find the dendrite point for the theta
            for theta in self.midpoint_list:
                # convert to radians
                theta *= (math.pi/180)
                # y goes from bottom up 
                x_zoom,y_zoom = r*math.cos(theta),-r*math.sin(theta)
                x,y,z = self.curr_point.point
                x,y,z = x_zoom+x,y_zoom+y,z
                if(x < 0): x = 0
                if(y < 0): y = 0
                if(x >= max_image_size): x = max_image_size-1
                if(y >= max_image_size): y = max_image_size-1
                # indices reversed row,col = y,x
                try:
                    z = numpy.argmax(self.img[y,x,z-d:z+d+1])-d+z
                except:
                    height = self.img.shape[2]
                    upper_lim = min(z+d,height)
                    lower_lim = max(z-d,0)
                    diff = z-lower_lim
                    z = (numpy.argmax(self.img[y,x,lower_lim:upper_lim+1]) 
                        - diff + z)
                self.auto_add_dendrite_points(x,y,z)


    def auto_add_dendrite_points(self,x,y,z):
        r = dt_point.RADIUS
        point = (x,y,z)
        if(not self.auto_repeated(dt_point(point))):
            point,center = (x,y,z),self.curr_point.point
            new_point = dt_point(point,center)
            self.dendrite_points.append(new_point)
            self.priority_queue.append(new_point)
            self.curr_point.add_connection(new_point)
            self.redraw_all()
            return True
        return False


    # helper function to check redundant points
    def auto_repeated(self,point):
        self.dist_limit = 6
        for dendrite_point in self.dendrite_points:
            if(point.dist(dendrite_point) < self.dist_limit):
                return True
        return False


    def auto_move(self):
        self.prev_point = copy.copy(self.curr_point)
        self.curr_point = self.priority_queue.pop()
        self.curr_point.add_connection(self.prev_point)
        self.on_classify()
        if(len(self.priority_queue) == 0):
            self.log(DendriteTrace.LINE)
            self.log('Done tracing the dendrite!')
            self.log('-Press the fix button to add missed branches')
            wx.MessageBox('Done tracing the dendrite!','Info',wx.OK)
            wx.MessageBox('Press the fix button to add missed branches',
                'Info',wx.OK)
        self.redraw_all()


    def auto_fix(self,event):
        self.image_panel.SetFocus()
        self.fix_mode = True
        # go to first point
        self.log('\t--> Fix mode selected')
        self.log(DendriteTrace.LINE)
        self.log('-Use up and down keys to navigate')
        self.log('-Press add to add a point')
        self.cursor = len(self.dendrite_points)-1
        self.curr_point = self.dendrite_points[self.cursor]
        self.redraw_all()


    def on_add(self,event):
        # act like manual mode
        self.log('-Use left and right to move around circle')
        self.log('-Use up and down to change the speed')
        self.log('-Press space to enter points')
        self.log('-Press enter to move to the next point')
        self.log('-Press trace to let the program do the rest')
        self.manual_mode = True
        self.image_panel.SetFocus()
        # make priority_queue nonempty
        self.priority_queue.append(self.curr_point)
        self.redraw_all()


def main():
    app = wx.App()
    DendriteTrace()
    app.MainLoop()


if __name__ == '__main__':
    main()