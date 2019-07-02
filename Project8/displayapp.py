# Skeleton File for CS 251 Project 3
# You can use this file or your file from Project 1
# Delete unnecessary code before handing in

import tkinter as tk
import math
import random
import numpy as np
import view
from tkinter import filedialog
import data
import analysis
# import pyscreenshot as ImageGrab

# create a class to build and manage the display
class DisplayApp:

    def __init__(self, width, height):
        self.pcaplot = False
        # create a tk object, which is the root window
        self.root = tk.Tk()

        # width and height of the window
        self.initDx = width
        self.initDy = height

        # set up the geometry for the window
        self.root.geometry("%dx%d+50+30" % (self.initDx, self.initDy))

        # set the title of the window
        self.root.title("Viewing Axes")

        # set the maximum size of the window for resizing
        self.root.maxsize(1024, 768)

        # bring the window to the front
        self.root.lift()

        # setup the menus
        self.buildMenus()

        # build the controls
        self.buildControls()

        # build the objects on the Canvas
        self.buildCanvas()

        # set up the key bindings
        self.setBindings()

        # Create a View object and set up the default parameters
        self.view = view.View()

        # Create the axes fields and build the axes
        self.endpoints = np.matrix([[0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1, 1]])
        self.lines = []
        self.line_of_fit = None
        self.headers = []
        self.buildAxes()
        
        # set up the application state
        self.objects = []
        self.data = None
        self.points = []

        # fields to hold the last values of button 1 motion
        self.b1mx = None
        self.b1my = None
        self.vrp = self.view.getVRP()
        self.prev_y = None
        self.base_click2 = []
        self.rot_x = None
        self.rot_y = None

        # matrix to hold the points to plot
        self.plot = None

        # Flags to determine whether or not the program should plot using some scale
        self.menuFlags = [False, False, False, False, False]

        self.regression_lines = []
        self.regression_endpoints = None

        # stores the values in the PCAlistbox
        self.PCAs = []
        self.pca_analysis = False 

        self.kmeans = []
        self.col_list = []
    def buildMenus(self):

        # create a new menu
        self.menu = tk.Menu(self.root)

        # set the root menu to our new menu
        self.root.config(menu=self.menu)

        # create a variable to hold the individual menus
        self.menulist = []

        # create a file menu
        filemenu = tk.Menu(self.menu)
        self.menu.add_cascade(label="File", menu=filemenu)
        self.menulist.append(filemenu)

        commandmenu = tk.Menu(self.menu)
        self.menu.add_cascade(label="Command", menu=commandmenu)
        self.menulist.append(commandmenu)
        # menu text for the elements
        menutext = [['Open - O', '-', 'Quit - Q', 'Save'],
                    ['Linear Regression in 2 Dimensions', 'Linear Regression in 3 Dimensions']]

        # menu callback functions
        menucmd = [[self.handleOpen, None, self.handleQuit, self.handleSave],
                   [self.handleLinearRegression, self.handleThreeDLinearRegression]]

        # build the menu elements and callbacks
        for i in range(len(self.menulist)):
            for j in range(len(menutext[i])):
                if menutext[i][j] != '-':
                    self.menulist[i].add_command(label=menutext[i][j], command=menucmd[i][j])
                else:
                    self.menulist[i].add_separator()

    # create the canvas object
    def buildCanvas(self):
        self.canvas = tk.Canvas(self.root, width=self.initDx, height=self.initDy)
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)
        return

    # build a frame and put controls in it
    def buildControls(self):
        # make a control frame
        self.cntlframe = tk.Frame(self.root)
        self.cntlframe.pack(side=tk.RIGHT, padx=2, pady=2, fill=tk.Y)

        sep = tk.Frame(self.root, height=self.initDy, width=2, bd=1, relief=tk.SUNKEN)
        sep.pack(side=tk.RIGHT, padx=2, pady=2, fill=tk.Y)

        # helpfully-named buttons
        self.buttons = []
        self.buttons.append(
            ('new window', tk.Button(self.cntlframe, text='Make a New Window', command=self.handleNewWindow, width=16)))
        self.buttons[-1][1].pack(side=tk.TOP)  # default side is top
        self.buttons.append(
            ('reset', tk.Button(self.cntlframe, text="Reset", command=self.handleResetButton, width=16)))
        self.buttons[-1][1].pack(side=tk.TOP)  # default side is top
        self.buttons.append(
            ('plot data', tk.Button(self.cntlframe, text="Plot Data", command=self.handlePlotData, width=16)))
        self.buttons[-1][1].pack(side=tk.TOP)  # default side is top
        self.buttons.append(
            ('PCA', tk.Button(self.cntlframe,text='PCA', command=self.handlePCA, width=16)))
        self.buttons[-1][1].pack(side=tk.TOP)  # default side is top
        self.buttons.append(
            ('PlotPCA', tk.Button(self.cntlframe,text='PlotPCA', command=self.handlePlotPCA, width=16)))
        self.buttons[-1][1].pack(side=tk.TOP)  # default side is top
        self.buttons.append(
            ('ShowEigenStuff', tk.Button(self.cntlframe,text='ShowEigenStuff', command=self.handleShowEigenStuff, width=16)))
        self.buttons[-1][1].pack(side=tk.TOP)  # default side is top
        self.buttons.append(
            ('K-Means', tk.Button(self.cntlframe,text='K-Means', command=self.handleKMeans, width=16)))
        self.buttons[-1][1].pack(side=tk.TOP)  # default side is top
        self.buttons.append(
            ('Euclidean', tk.Button(self.cntlframe,text='Euclidean', command=self.handleEuclidean, width=16)))
        self.buttons[-1][1].pack(side=tk.TOP)  # default side is top
        self.buttons.append(
            ('Manhattan', tk.Button(self.cntlframe,text='Manhattan', command=self.handleManhattan, width=16)))
        self.buttons[-1][1].pack(side=tk.TOP)  # default side is top
        
        self.buttons.append(
            ('Show Means', tk.Button(self.cntlframe,text='Show Means', command=self.handleShowKMeans, width=16)))
        self.buttons[-1][1].pack(side=tk.TOP)  # default side is top
        
        
        # We will use this in handleChooseAxes() to
        self.optionMenus = []

        self.PCAlistbox = tk.Listbox(self.cntlframe, selectmode=tk.MULTIPLE)
        self.PCAlistbox.pack(side=tk.TOP)
        
        self.cols = tk.Listbox(self.cntlframe, selectmode=tk.EXTENDED)
        self.cols.pack(side=tk.TOP)
        
        self.manhattan = False
        return

    def handleEuclidean(self):
        self.manhattan = False

    def handleManhattan(self):
        self.manhattan = True

    def handleSave(self):
        nd = NameDialog(self.root)
        filename = nd.get_name()
        print(type(self.data))
        self.data.write(filename)

    def handleKMeans(self):
        #this line cannot go after the dialogs are created because if the user
        #types something into the dialogs, it will deselect the columns
        self.headers = self.cols.curselection()
        nd = NameDialog(self.root)
        name = nd.get_name()
        kd = KDialog(self.root)
        self.k = kd.get_k()
        for i in range(len(self.headers)):
            if self.headers[i] == 'None':
                del self.headers[i]
        print("headers to cluster:", self.headers)
        codebook, codes, errors, quality = analysis.kmeans(self.data, self.headers, self.k, whiten=False, manhattan=self.manhattan)
        print("MDL:",quality)
        self.kmeans.append([codebook, codes, errors, quality])
        if name == "Default":
            self.PCAlistbox.insert(tk.END, "K-Means" + str(self.PCAlistbox.size()))
        else:
            self.PCAlistbox.insert(tk.END, name)

    #passes all of the relevant stuff to the EigenDialog constructor
    def handleShowEigenStuff(self):
        pca = self.PCAs[self.PCAlistbox.curselection()[0]]
        print("PCA:",pca)
        ed = EigenDialog(self.root, pca.get_eigenvectors(), pca.get_eigenvalues(), pca.get_original_headers()) 
        

    def handleShowKMeans(self):
        km = self.kmeans[self.PCAlistbox.curselection()[0]][0]
        ed = MeansDialog(self.root, self.k, self.kmeans[self.PCAlistbox.curselection()[0]][0], self.headers, [self.col_list[i] for i in self.headers])#km.get_headers()) 

    #does some stuff with headers and then calls buildPoints
    def handlePlotPCA(self):
        self.pcaplot = True
        tmp = []
        for header in self.headers:
            tmp.append(header)
        self.headers = tmp
        
        while len(self.headers) < 6:
            self.headers.append('None')
        for i in range(5):
            if self.headers[i] != 'None':
                self.menuFlags[i] = True
            else:
                self.menuFlags[i] = False
        index = self.PCAlistbox.curselection()[0]
        self.headers = self.PCAs[index].get_headers()
        self.buildPoints(self.headers)

    #handles the user pressing the PCA button
    def handlePCA(self):
        #getting the name of the analysis
        nd = NameDialog(self.root)
        name = nd.get_name()
        
        self.pca_analysis = True
        
        self.headers = self.cols.curselection()
        #get rid of anything we don't care about
        for i in range(len(self.headers)):
            if self.headers[i] == 'None':
                del self.headers[i]

        #making the pca object
        new_pca = analysis.pca(self.data, self.headers, prenormalize=True)
        self.PCAs.append(new_pca)

        #print out all of the values
        print("\n\nValues from PCA:")
        print("\neigenvalues:\n",new_pca.get_eigenvalues())
        print("\neigenvectors:\n",new_pca.get_eigenvectors())
        print("\nmeans:\n",new_pca.get_original_means())
        print("\nheaders:\n",new_pca.get_original_headers())
        
        #naming the pca and inserting it
        if name == "Default":
            self.PCAlistbox.insert(tk.END, "PCA" + str(self.PCAlistbox.size()))
        else:self.PCAlistbox.insert(tk.END, name)
    
    def handleThreeDLinearRegression(self):
        # get the dependent and indepedendent variables
        td = ThreeDDialog(self.root, self.data.get_headers())
        dependent_variable = td.get_dep_val()
        independent_variables = td.get_ind_vals()

        # clear the existing points from the window
        for point in self.points:
            self.canvas.delete(point)
        self.points = []

        for line in self.regression_lines:
            self.canvas.delete(line)
        self.regression_lines = []
        self.regression_endpoints = None

        self.view.reset()
        self.build_3d_linear_regression(independent_variables, dependent_variable)
        self.updateAxes()
        self.updateFits()

    def build_3d_linear_regression(self, independent_variables, dependent_variable):
        self.plot = analysis.normalize_columns_separately([independent_variables[0], independent_variables[1],
                                                           dependent_variable], self.data)

        # self.plot = self.data.limit_columns([independent_variable, dependent_variable])
        self.plot = np.hstack((self.plot, np.ones((self.plot.shape[0], 1))))

        # build the view matrix and transform the points
        vtm = self.view.build()
        pts = self.plot * vtm  # (vtm * self.plot.T).T

        # initialize self.size so that our movement functions don't break
        self.size = []
        # make a graphical point for each data point
        for i in range(len(pts)):
            self.size.append(1)
            x = pts[i, 0]
            y = pts[i, 1]
            pt = self.canvas.create_oval(int(x - 1), int(y - 1),
                                         int(x + 1), int(y + 1),
                                         fill="black", outline='')
            self.points.append(pt)

        linres = analysis.linear_regression(self.data, independent_variables, dependent_variable)
        slope0 = linres[0]
        slope1 = linres[1]

        intercept = linres[2]
        rvalue = linres[4]

        xmin = analysis.data_range([independent_variables[0]], self.data)[0][0]
        xmax = analysis.data_range([independent_variables[0]], self.data)[0][1]
        ymin = analysis.data_range([independent_variables[1]], self.data)[0][0]
        ymax = analysis.data_range([independent_variables[1]], self.data)[0][1]
        zmin = analysis.data_range([dependent_variable], self.data)[0][0]
        zmax = analysis.data_range([dependent_variable], self.data)[0][1]

        xends = [0.0, 1.0]
        yends = [((xmin * slope0[0,0] + intercept[0,0]) - ymin) / (ymax - ymin),
                 ((xmax * slope0[0,0] + intercept[0,0]) - ymin) / (ymax - ymin)]
        zends = [((xmin * slope1[0,0] + intercept[0,0]) - zmin) / (zmax - zmin),
                 ((xmax * slope1[0,0] + intercept[0,0]) - zmin) / (zmax - zmin)]

        self.regression_endpoints = np.matrix([[0.0, 1.0],
                                               [yends[0], yends[1]],
                                               [zends[0], zends[1]],
                                               [1, 1]])

        print("self.regression_endpoints", self.regression_endpoints)
        self.line_of_fit = (self.canvas.create_line(self.regression_endpoints[0,0],self.regression_endpoints[1,0],
                                            self.regression_endpoints[0,1],self.regression_endpoints[1,1], fill="red"))

        self.regression_lines.append(self.line_of_fit)
        self.fit_label = tk.Label(self.canvas, text="slope0: " + str(slope0[0,0]) +
                                                    "\nslope1: " + str(slope1[0,0]) +
                                                    "\nIntercept: " + str(intercept[0,0]) +
                                                    "\nR-value: " + str(rvalue))
        self.fit_label.place(x=self.regression_endpoints[0, 1], y=self.regression_endpoints[1, 1])
        self.updateAxes()
        self.updateFits()
        self.updatePoints()

    def handleLinearRegression(self):
        #get the dependent and indepedendent variables
        md = ModalDialog(self.root, self.data.get_headers())
        dependent_variable = md.get_dep_val()
        independent_variable = md.get_ind_val()

        #clear the existing points from the window
        for point in self.points:
            self.canvas.delete(point)
        self.points = []

        for line in self.regression_lines:
            self.canvas.delete(line)
        self.regression_lines = []
        self.regression_endpoints = None

        self.view.reset()
        self.build_linear_regression(independent_variable, dependent_variable)
        self.updateAxes()
        self.updateFits()

    def build_linear_regression(self, independent_variable, dependent_variable):

        #initialize the matrix of data we want to do a linear regression on
        self.plot = analysis.normalize_columns_separately([independent_variable,dependent_variable],self.data)
        # self.plot = self.data.limit_columns([independent_variable, dependent_variable])
        self.plot = np.hstack((self.plot, np.zeros((self.plot.shape[0], 1))))
        self.plot = np.hstack((self.plot, np.ones((self.plot.shape[0], 1))))

        #build the view matrix and transform the points
        vtm = self.view.build()
        pts = self.plot * vtm #(vtm * self.plot.T).T

        #initialize self.size so that our movement functions don't break
        self.size = []
        #make a graphical point for each data point
        for i in range(len(pts)):
            self.size.append(3)
            x = pts[i, 0]
            y = pts[i, 1]
            if self.vshape.get() == 'Triangle':
                pt = self.canvas.create_polygon(int(x - self.size[i]), int(y + self.size[i]),
                                                    int(x + self.size[i]), int(y + self.size[i]), int(x),
                                                    int(y - self.size[i]), fill='black', outline='')

            elif self.vshape.get() == 'Pentagon':
                print("pentagon")
                pt = self.canvas.create_polygon(
                                        (int(x - self.size[i]), int(y), int(x), int(y + self.size[i]),
                                         int(x + self.size[i]), int(y), int(x + self.size[i]),
                                         int(y - self.size[i]), int(x - self.size[i]), int(y - self.size[i])))

            elif self.vshape.get() == 'Circle':
                pt = self.canvas.create_oval(int(x - self.size[i]), int(y - self.size[i]),
                                                    int(x + self.size[i]), int(y + self.size[i]), fill='black', outline='')

            elif self.vshape.get() == 'Rectangle':
                pt = self.canvas.create_rectangle(int(x - self.size[i]), int(y - self.size[i]),
                                        int(x + self.size[i]), int(y + self.size[i]), fill='black', outline='')

            else:
                pt = self.canvas.create_arc(int(x - self.size[i]), int(y - self.size[i]),
                                        int(x + self.size[i]), int(y + self.size[i]),fill='black', outline='')

            # pt = self.canvas.create_oval(int(x - 1), int(y - 1),
            #                              int(x + 1), int(y + 1),
            #                              fill="black", outline='')
            self.points.append(pt)

        linres = analysis.single_linear_regression(self.data, independent_variable, dependent_variable)
        slope = linres[0]
        intercept = linres[1]
        rvalue = linres[2]
        pvalue = linres[3]
        stderr = linres[4]
        xmin = linres[5]
        xmax = linres[6]
        ymin = linres[7]
        ymax = linres[8]

        xends = [0.0, 1.0]
        yends = [((xmin * slope + intercept) - ymin)/(ymax - ymin), ((xmax * slope + intercept) - ymin)/(ymax - ymin)]


        self.regression_endpoints = np.matrix([[0.0,1.0],
                               [yends[0], yends[1]],
                               [0,0],
                               [1,1]])

        self.line_of_fit = (self.canvas.create_line(self.regression_endpoints[0,0],self.regression_endpoints[1,0],
                                                  self.regression_endpoints[0,1],self.regression_endpoints[1,1], fill="red"))

        self.regression_lines.append(self.line_of_fit)
        self.fit_label = tk.Label(self.canvas, text="slope: " + str(slope) +
                                                    "\nIntercept: " + str(intercept) +
                                                    "\nR-value: " + str(rvalue))
        self.fit_label.place(x=self.regression_endpoints[0,1], y=self.regression_endpoints[1,1])
        self.updateAxes()
        self.updateFits()
        self.updatePoints()

    def handleNewWindow(self):
        newdapp = DisplayApp(1200, 1200)

    # create the axis line objects in their default location
    def buildAxes(self):

        vtm = self.view.build()
        pts = vtm * self.endpoints
        # Extension 3 puts colors on the axes so that the user can tell them apart
        self.lines.append(self.canvas.create_line(pts[0, 0], pts[1, 0], pts[0, 1], pts[1, 1], fill="blue"))
        self.lines.append(self.canvas.create_line(pts[0, 2], pts[1, 2], pts[0, 3], pts[1, 3], fill="red"))
        self.lines.append(self.canvas.create_line(pts[0, 4], pts[1, 4], pts[0, 5], pts[1, 5], fill="green"))

        # Extension 3 labels the axes
        self.xlabel = tk.Label(self.canvas, text='x')
        self.xlabel.place(x=pts[0, 1], y=pts[1, 1])
        self.ylabel = tk.Label(self.canvas, text='y')
        self.ylabel.place(x=pts[0, 3], y=pts[1, 3])
        self.zlabel = tk.Label(self.canvas, text='z')
        self.zlabel.place(x=pts[0, 5], y=pts[1, 5])

    def updateFits(self):
        if self.line_of_fit is not None:
            vtm = self.view.build()
            pts = vtm * self.regression_endpoints
            self.canvas.coords(self.line_of_fit, [pts[0,0], pts[1,0], pts[0,1], pts[1,1]])
        else:
            return
    # modify the endpoints of the axes to their new location
    def updateAxes(self):
        vtm = self.view.build()
        pts = vtm * self.endpoints
        for i in range(len(self.lines)):
            self.canvas.coords(self.lines[i], [pts[0, i * 2], pts[1, i * 2], pts[0, i * 2 + 1], pts[1, i * 2 + 1]])

        self.xlabel.destroy()
        self.ylabel.destroy()
        self.zlabel.destroy()
        if len(self.headers) > 2:
            self.xlabel = tk.Label(self.canvas, text=self.headers[0])
            self.ylabel = tk.Label(self.canvas, text=self.headers[1])
            self.zlabel = tk.Label(self.canvas, text=self.headers[2])
        else:
            self.xlabel = tk.Label(self.canvas, text='x')
            self.ylabel = tk.Label(self.canvas, text='y')
            self.zlabel = tk.Label(self.canvas, text='z')

        self.xlabel.place(x=pts[0, 1], y=pts[1, 1])
        self.ylabel.place(x=pts[0, 3], y=pts[1, 3])
        self.zlabel.place(x=pts[0, 5], y=pts[1, 5])

    def setBindings(self):
        self.canvas.bind('<Button-1>', self.handleButton1)
        self.canvas.bind('<Button-2>', self.handleButton2)
        self.canvas.bind('<Control-Button-1>', self.handleButton2)
        self.canvas.bind('<Button-3>', self.handleButton3)
        self.canvas.bind('<B1-Motion>', self.handleButton1Motion)
        self.canvas.bind('<B2-Motion>', self.handleButton2Motion)
        self.canvas.bind('<Control-B1-Motion>', self.handleButton2Motion)
        self.canvas.bind('<B3-Motion>', self.handleButton3Motion)
        self.root.bind('<Control-q>', self.handleQuit)
        self.root.bind('<Control-o>', self.handleModO)
        # self.root.bind('<Control-l>', self.handleGrab)
        self.root.bind('<Delete>', self.handleDelete)
        # Extension 2 lets the user use more intuitive controls to scale
        self.canvas.bind('<Shift-Button-1>', self.handleButton3)
        self.canvas.bind('<Shift-B1-Motion>', self.handleButton3Motion)
        return

    def handleDelete(self, event):
        #a list of everything that can be selected - i can't select more than one thing on my machine
        #but i think it might work on Mac?
        selections = [self.PCAlistbox.curselection()[i] for i in range(len(self.PCAlistbox.curselection()))]
        for i in range(len(selections)):
            self.PCAlistbox.delete(selections[i])
            del self.PCAs[selections[i]]
         
    # def handleGrab(self, file_path='image.jpg'):
    #     box = (self.root.winfo_rootx(), self.root.winfo_rooty(), self.root.winfo_rootx() +  1.28 * self.root.winfo_width(),
    #            self.root.winfo_rooty() + 1.28 * self.root.winfo_height())
    #     # grab = ImageGrab.grab(bbox=box)
    #     # grab.save('img.jpg')
    #     print("saved image")

    def buildPoints(self, headers):
        print("\n\nheaders for buildPoints:",headers)
        # delete existting canvas objects used for plotting data
        for point in self.points:
            self.canvas.delete(point)
        self.points = []
        if self.pcaplot:
            self.plot_data = analysis.normalize_columns_separately(headers, self.PCAs[self.PCAlistbox.curselection()[0]])
        else:
            self.plot_data = analysis.normalize_columns_separately(headers, self.data)
        self.plot = self.plot_data[:, :2]
        z_flag = False
        if self.menuFlags[2]:
            self.plot = np.hstack((self.plot, self.plot_data[:, 2]))
            z_flag = True
        else:
            self.plot = np.hstack((self.plot, np.zeros((len(self.plot), 1))))

        size_flag = False
        if self.menuFlags[3]:
            size_flag = True
            if z_flag:
                self.size = self.plot_data[:, 3]
            else:
                self.size = self.plot_data[:, 2]
        else:
            self.size = np.ones((len(self.plot), 1))
        self.size = 3 * self.size + 1

        if not self.data.get_kmeans():

            if self.menuFlags[4]:
                if z_flag and size_flag:
                    color = self.plot_data[:, 4]
                elif (z_flag and not size_flag) or (not z_flag and size_flag):
                    color = self.plot_data[:, 3]
                else:
                    color = self.plot_data[:, 2]

                self.green = -255 * color + 255
                self.red = 255 * color
            else:
                self.green = np.ones((len(self.plot), 1))
                self.red = np.ones((len(self.plot), 1))

        # homogeneous coordinate
        self.plot = np.hstack((self.plot, np.ones((self.plot.shape[0], 1))))

        # make a vtm so the points aren't tiny
        vtm = self.view.build()

        # put the points through the vtm
        pts = (vtm * self.plot.T).T
        # loop over the points, drawing each one
        if not self.data.get_kmeans():
            for i in range(len(pts)):
                x = pts[i, 0]
                y = pts[i, 1]

                # Extension 1 gives the user the capability to use different shapes
                if self.vshape.get() == 'Circle':
                    pt = self.canvas.create_oval(int(x - self.size[i]), int(y - self.size[i]),
                                                 int(x + self.size[i]), int(y + self.size[i]),
                                                 fill="#%02x%02x%02x" % (int(self.red[i]),
                                                                         int(self.green[i]), 0), outline='')
                elif self.vshape.get() == 'Rectangle':
                    pt = self.canvas.create_rectangle(int(x - self.size[i]), int(y - self.size[i]),
                                                      int(x + self.size[i]), int(y + self.size[i]),
                                                      fill="#%02x%02x%02x" % (int(self.red[i]),
                                                                              int(self.green[i]), 0), outline='')
                elif self.vshape.get() == 'Triangle':
                    pt = self.canvas.create_polygon(int(x - self.size[i]), int(y + self.size[i]), int(x + self.size[i]),
                                                    int(y + self.size[i]), int(x), int(y - self.size[i]))
                elif self.vshape.get() == 'Pentagon':
                    pt = self.canvas.create_polygon(int(x - self.size[i]), int(y), int(x), int(y + self.size[i]),
                                                    int(x + self.size[i]),
                                                    int(y), int(x + self.size[i]), int(y - self.size[i]),
                                                    int(x - self.size[i]), int(y - self.size[i]))
                else:
                    pt = self.canvas.create_arc(int(x - self.size[i]), int(y - self.size[i]),
                                                int(x + self.size[i]), int(y + self.size[i]),
                                                fill="#%02x%02x%02x" % (int(self.red[i]),
                                                                        int(self.green[i]), 0), outline='')
                self.points.append(pt)

        #if we are doing a kmeans analysis
        else:

            for i in range(len(pts)):
                x = pts[i, 0]
                y = pts[i, 1]
                color = self.kmeans[self.PCAlistbox.curselection()[0]][1][i,0]
                print("color:",color)
                if color == 0:
                    rgb = "#%02x%02x%02x" % (255,0,0)
                elif color == 1:
                    rgb = "#%02x%02x%02x" % (255,111,0)
                elif color == 2:
                    rgb = "#%02x%02x%02x" % (255,221,0)
                elif color == 3:
                    rgb = "#%02x%02x%02x" % (178,255,0)
                elif color == 4:    
                    rgb = "#%02x%02x%02x" % (0,255,153)
                elif color == 5:
                    rgb = "#%02x%02x%02x" % (0,247,255)
                elif color == 6:
                    rgb = "#%02x%02x%02x" % (0,136,255)
                elif color == 7:
                    rgb = "#%02x%02x%02x" % (0,25,255)
                elif color == 8:
                    rgb = "#%02x%02x%02x" % (196,0,255)
                elif color == 9:
                    rgb = "#%02x%02x%02x" % (255,0,94)
                else:
                    rgb = "#%02x%02x%02x" % (random.randint(0,255), random.randint(0,255), random.randint(0,255))

                if self.vshape.get() == 'Circle':
                    pt = self.canvas.create_oval(int(x - self.size[i]), int(y - self.size[i]),
                                                 int(x + self.size[i]), int(y + self.size[i]),
                                                 fill=rgb, outline='')
                elif self.vshape.get() == 'Rectangle':
                    pt = self.canvas.create_rectangle(int(x - self.size[i]), int(y - self.size[i]),
                                                      int(x + self.size[i]), int(y + self.size[i]),
                                                      fill=rgb, outline='')
                elif self.vshape.get() == 'Triangle':
                    pt = self.canvas.create_polygon(int(x - self.size[i]), int(y + self.size[i]), int(x + self.size[i]),
                                                    int(y + self.size[i]), int(x), int(y - self.size[i]), fill=rgb, outline='')
                elif self.vshape.get() == 'Pentagon':
                    pt = self.canvas.create_polygon(int(x - self.size[i]), int(y), int(x), int(y + self.size[i]),
                                                    int(x + self.size[i]),
                                                    int(y), int(x + self.size[i]), int(y - self.size[i]),
                                                    int(x - self.size[i]), int(y - self.size[i]), fill=rgb, outline='')
                else:
                    pt = self.canvas.create_arc(int(x - self.size[i]), int(y - self.size[i]),
                                                int(x + self.size[i]), int(y + self.size[i]),
                                                fill=rgb, outline='')

                # put the point object into self.points
                self.points.append(pt)
        return

    def updatePoints(self):
        # don't do anything if there aren't any points
        if len(self.points) == 0:
            return
        # make a new vtm and use it to transform the points
        vtm = self.view.build()
        pts = (vtm * self.plot.T).T

        # radius of points
        dx = 1
        dy = 1
        # update the coordinates of each point
        for i in range(pts.shape[0]):
            for j in range(pts.shape[1]):
                x = pts[i, 0]
                y = pts[i, 1]
                if self.vshape.get() == 'Triangle':
                    self.canvas.coords(self.points[i], (int(x - self.size[i]), int(y + self.size[i]),
                                                        int(x + self.size[i]), int(y + self.size[i]), int(x),
                                                        int(y - self.size[i])))
                elif self.vshape.get() == 'Pentagon':
                    pt = self.canvas.coords(self.points[i],
                                            (int(x - self.size[i]), int(y), int(x), int(y + self.size[i]),
                                             int(x + self.size[i]), int(y), int(x + self.size[i]),
                                             int(y - self.size[i]), int(x - self.size[i]), int(y - self.size[i])))
                else:
                    self.canvas.coords(self.points[i], (int(x - self.size[i]), int(y - self.size[i]),
                                                        int(x + self.size[i]), int(y + self.size[i])))
        print("self.vx", self.vx.get())
        return

    def handlePlotData(self):
        self.pcaplot = False
        # The user will hit plot after they have selected which axes to plot (hopefully)

        self.headers = [self.vx.get(), self.vy.get(), self.vz.get(), self.vcolor.get(), self.vsize.get()]
        
        # set the menuFlags to True if the user wants to plot it
        for i in range(5):
            if self.headers[i] != 'None':
                self.menuFlags[i] = True
        print("self.menuFlags", self.menuFlags)
        # get rid of None headers with this loop
        i = 4
        while i >= 0:
            if self.headers[i] == 'None':
                del self.headers[i]
            i -= 1

        # put the points on the screen
        self.buildPoints(self.headers)

    # returns the first three headers of the data
    def handleChooseAxes(self):
        # put data headers into a list
        self.headers = self.data.get_headers()
        for header in self.headers:
            self.cols.insert(tk.END,header)
            self.col_list.append(header)
        choices = tuple(self.headers)
        # we need the StringVar to tell which option is selected
        self.vx = tk.StringVar(self.root)
        # set default selected option to the first item in headers
        self.vx.set(self.headers[0])
        # * dereferences choices so we can pass in an arbitrary number of arguments
        xfeat = tk.OptionMenu(self.cntlframe, self.vx, *choices)
        # put it into the cntlframe
        xfeat.pack(side=tk.TOP)

        self.vy = tk.StringVar(self.root)
        self.vy.set(self.headers[1])
        yfeat = tk.OptionMenu(self.cntlframe, self.vy, *choices)
        yfeat.pack(side=tk.TOP)

        self.headers.append('None')
        choices = tuple(self.headers)
        self.vz = tk.StringVar(self.root)
        self.vz.set(None)
        zfeat = tk.OptionMenu(self.cntlframe, self.vz, *choices)
        zfeat.pack(side=tk.TOP)

        self.vcolor = tk.StringVar(self.root)
        self.vcolor.set(None)
        colorfeat = tk.OptionMenu(self.cntlframe, self.vcolor, *choices)
        colorfeat.pack(side=tk.TOP)

        self.vsize = tk.StringVar(self.root)
        self.vsize.set(None)
        sizefeat = tk.OptionMenu(self.cntlframe, self.vsize, *choices)
        sizefeat.pack(side=tk.TOP)

        shapes = ('Circle', 'Rectangle', 'Triangle', 'Pentagon', 'Arc')
        self.vshape = tk.StringVar(self.root)
        self.vshape.set('Circle')
        shape_option = tk.OptionMenu(self.cntlframe, self.vshape, *shapes)
        shape_option.pack(side=tk.TOP)

        # if self.pca_analysis:
        self.feats = [xfeat, yfeat, zfeat, colorfeat, sizefeat]
        #     stringvars = [self.vx, self.vy, self.vz, self.vcolor, self.vsize]
        #     PCAChoices = ['PCA' + str(i) for i in range(len(self.headers))]
        #     for i in range(len(feats)):
        #         feats[i] = tk.OptionMenu(self.cntlframe, stringvars[i], (PCAChoices[i], None))
        #         feats[i].pack(side=tk.TOP)
        return self.headers

    def handleOpen(self):
        fn = filedialog.askopenfilename(parent=self.root,
                                        title='Choose a data file', initialdir='.')
        self.data = data.Data()
        self.data.read(fn)
        self.handleChooseAxes()
        self.updateAxes()

        print('handleOpen')

    def handleModO(self, event):
        self.handleOpen()

    def handleQuit(self, event=None):
        print('Terminating')
        self.root.destroy()

    # Extension 4 implements reset
    def handleResetButton(self):
        self.view.reset()
        self.updateAxes()
        self.updateFits()
        self.updatePoints()
        print('handling reset button')

    def handleButton1(self, event):
        print('handle button 1: %d %d' % (event.x, event.y))
        self.b1mx = event.x
        self.b1my = event.y
        self.vrp = self.view.getVRP()

    # rotation
    def handleButton2(self, event):
        self.base_click2 = (event.x, event.y)
        # self.base_click2.append(event.y)
        self.clone = self.view.clone()
        print('handle button 2: %d %d' % (event.x, event.y))

    # scaling
    def handleButton3(self, event):
        self.button3_location = [event.x, event.y]
        self.original_extent = self.view.getExtent()
        print('handle button 3: %d %d' % (event.x, event.y))

    # translation
    def handleButton1Motion(self, event):
        # get the changes in x and y
        if self.b1mx == None:
            self.b1mx = event.x
        if self.b1my == None:
            self.b1my = event.y
        dx = event.x - self.b1mx
        dy = event.y - self.b1my

        # divide by screen size
        screen = self.view.getScreen()
        dx /= screen[0]
        dy /= screen[1]

        # multiply by extent
        extent = self.view.getExtent()
        dx *= extent[0, 0]
        dy *= extent[0, 1]

        # update VRP
        self.view.setVRP(self.vrp + dx * self.view.getU() + dy * self.view.getVUP())
        # print(self.vrp)
        self.updateAxes()
        self.updateFits()
        self.updatePoints()
        print('handle button 1 motion: %d %d' % (event.x, event.y))

    # Rotation
    def handleButton2Motion(self, event):
        dx = (self.base_click2[0] - event.x) * math.pi / 200
        dy = (self.base_click2[1] - event.y) * math.pi / 200
        self.view = self.clone.clone()
        self.view.rotateVRC(dx, dy)
        self.updateAxes()
        self.updateFits()
        self.updatePoints()
        print('handle button 2 motion: %d %d' % (event.x, event.y))

    # Scaling
    def handleButton3Motion(self, event):
        # see if this is the first time through this method
        # scale gets the difference between the prev_y and new y
        scale = self.button3_location[1] - event.y
        # update prev_y
        self.button3_location[1] = event.y

        # doing this sets the default to 1 and makes it so that moving
        # the mouse a lot doesn't affect it way too much
        scale += 100
        scale /= 100

        # keep the scale between 3 and 0.1
        if scale > 3.0:
            scale = 3.0
        elif scale < 0.1:
            scale = 0.1

        self.view.setExtent((1 / scale) * self.view.getExtent())
        self.updateAxes()
        self.updateFits()
        self.updatePoints()

        print('handle button 3 motion: %d %d' % (event.x, event.y))

    def main(self):
        print('Entering main loop')
        self.root.mainloop()

class Dialog(tk.Toplevel):
    # I did not write this class, it was provided via a link from the project page
    def __init__(self, parent, title=None):
        tk.Toplevel.__init__(self, parent)
        self.transient(parent)
        if title:
            self.title(title)
        self.parent = parent
        self.result = None
        body = tk.Frame(self)
        self.initial_focus = self.body(body)
        body.pack(padx=5, pady=5)
        self.buttonbox()
        # self.grab_set()
        if not self.initial_focus:
            self.initial_focus = self
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.geometry("+%d+%d" % (parent.winfo_rootx() + 50,
                                  parent.winfo_rooty() + 50))
        self.initial_focus.focus_set()
        self.wait_window(self)
    # construction hooks

    def body(self, master):
        # create dialog body.  return widget that should have
        # initial focus.  this method should be overridden
        return self

    def buttonbox(self):
        # add standard button box. override if you don't want the
        # standard buttons
        box = tk.Frame(self)
        w = tk.Button(box, text="OK", width=10, command=self.ok, default=tk.ACTIVE)
        w.pack(side=tk.LEFT, padx=5, pady=5)
        w = tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=tk.LEFT, padx=5, pady=5)
        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)
        box.pack()

    #
    # standard button semantics

    def ok(self, event=None):
        if not self.validate():
            self.initial_focus.focus_set()  # put focus back
            return
        self.withdraw()
        self.update_idletasks()
        self.apply()
        self.cancel()

    def cancel(self, event=None):
        # put focus back to the parent window
        self.parent.focus_set()
        self.destroy()
        self.userCancelled()

    # command hooks

    def userCancelled(self):
        pass  # override

    def validate(self):
        return 1  # override

    def apply(self):
        pass  # override

class ModalDialog(Dialog):

    def __init__(self, parent, headers):
        # initialize a new field to store the value the user types
        self.ind_val = tk.StringVar()
        self.dep_val = tk.StringVar()
        self.headers = headers

        # call parent constructor
        Dialog.__init__(self, parent)


    def body(self, master):
        tk.Label(master, text="Independent Variable:").grid(row=0)
        tk.Label(master, text="Dependent Variable:").grid(row=1)
        self.ind = tk.StringVar(master)
        self.ind.set(self.headers[0])

        self.ind_var = tk.OptionMenu(master, self.ind, *self.headers) #Entry(master)
        self.ind_var.grid(row=0, column=1)

        self.dep= tk.StringVar(self.master)
        self.dep.set(self.headers[1])

        self.dep_var = tk.OptionMenu(master, self.dep, *self.headers)  # Entry(master)
        self.dep_var.grid(row=1, column=1)

        return self.ind_var  # initial focus

    def apply(self):
        # put the new value in self.val
        self.ind_val = self.ind.get()
        self.dep_val = self.dep.get()

    # make sure that the input can be cast to an integer
    def validate(self):
        # try:
        #     self.ind_val = int(self.ind.get())
        # except ValueError:
        #     print("Could not convert to an integer.")
        #     return False
        # if self.ind_val > 1000 or self.ind_val < 0:
        #     print("Please put a value between 0 and 1000")
        #     return False
        return True

    # getter for self.val
    def get_ind_val(self):
        return self.ind_val

    def get_dep_val(self):
        return self.dep_val

    def userCancelled(self):
        # called by the cancel method, which only the user can
        # invoke, so anytime we're in here, the user has cancelled
        # so all we have to do is return true
        return False

class ThreeDDialog(Dialog):

    def __init__(self, parent, headers):
        # initialize a new field to store the value the user types
        self.ind_val0 = tk.StringVar()
        self.ind_val1 = tk.StringVar()
        self.dep_val = tk.StringVar()
        self.headers = headers

        # call parent constructor
        Dialog.__init__(self, parent)


    def body(self, master):
        tk.Label(master, text="Independent Variable 0 :").grid(row=0)
        tk.Label(master, text="Independent Variable 1 :").grid(row=1)
        tk.Label(master, text="Dependent Variable:").grid(row=2)
        self.ind0 = tk.StringVar(master)
        self.ind0.set(self.headers[0])

        self.ind_var0 = tk.OptionMenu(master, self.ind0, *self.headers) #Entry(master)
        self.ind_var0.grid(row=0, column=1)

        self.ind1 = tk.StringVar(master)
        self.ind1.set(self.headers[1])

        self.ind_var1 = tk.OptionMenu(master, self.ind1, *self.headers) #Entry(master)
        self.ind_var1.grid(row=1, column=1)

        self.dep= tk.StringVar(self.master)
        self.dep.set(self.headers[2])

        self.dep_var = tk.OptionMenu(master, self.dep, *self.headers)  # Entry(master)
        self.dep_var.grid(row=2, column=1)

        return self.ind_var0  # initial focus

    def apply(self):
        # put the new value in self.val
        self.ind_val0 = self.ind0.get()
        self.ind_val1 = self.ind1.get()
        self.dep_val = self.dep.get()

    # make sure that the input can be cast to an integer
    def validate(self):
        return True

    # getter for self.val
    def get_ind_vals(self):
        return [self.ind_val0, self.ind_val1]

    def get_dep_val(self):
        return self.dep_val

    def userCancelled(self):
        # called by the cancel method, which only the user can
        # invoke, so anytime we're in here, the user has cancelled
        # so all we have to do is return true
        return False

class EigenDialog(Dialog):

    def __init__(self, parent, eigenvectors, eigenvalues, headers):
        # initialize a new field to store the value the user types
        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues
        self.eigenval_sum = sum(eigenvalues)
        self.headers = headers
        # call parent constructor
        Dialog.__init__(self, parent)


    def body(self, master):
        tk.Label(master, text="E-vec").grid(row=0, column=0)
        tk.Label(master, text="E-val").grid(row=0, column=1)
        tk.Label(master, text="Cumulative").grid(row=0, column=2)
        for i in range(len(self.headers)):
            #the names of the columns
            tk.Label(master, text=self.headers[i]).grid(row=0, column=i+3)
            #the name of the eigenvector
            tk.Label(master, text="PCA" + str(i)).grid(row=i+1, column=0)
            #the first four digits of the eigenvalue
            tk.Label(master, text=str(self.eigenvalues[i])[0:5]).grid(row=i+1, column=1)
            #the cumulative percentage of the variance accounted for
            tk.Label(master, text=str(sum(self.eigenvalues[:i+1])/self.eigenval_sum)[:5]).grid(row=i+1, column=2)
            #put the values in the eigenvectors into the right spots
            for j in range(len(self.headers)):
                tk.Label(master, text=str(self.eigenvectors[i][0,j])[:5]).grid(row=i+1, column=j+3)

        return   # initial focus

    def apply(self):
        pass
        
    # make sure that the input can be cast to an integer
    def validate(self):
        return True

    def userCancelled(self):
        return False

class NameDialog(Dialog):

    def __init__(self, parent):
        # initialize a new field to store the value the user types
        self.name = tk.StringVar()
        # call parent constructor
        Dialog.__init__(self, parent)


    def body(self, master):
        tk.Label(master, text="Name of your analysis: ")
        e = tk.Entry(master, textvariable=self.name)
        e.pack()
        self.name.set("Default")
        return   # initial focus

    def apply(self):
        pass
        
    # make sure that the input can be cast to an integer
    def validate(self):
        return True

    def userCancelled(self):
        return False

    def get_name(self):
        return self.name.get()

class KDialog(Dialog):

    def __init__(self, parent):
        # initialize a new field to store the value the user types
        self.k = tk.StringVar()
        # call parent constructor
        Dialog.__init__(self, parent)


    def body(self, master):
        tk.Label(master, text="Number of clusters: ")
        e = tk.Entry(master, textvariable=self.k)
        e.pack()
        self.k.set("10")
        return   # initial focus

    def apply(self):
        pass
        
    # make sure that the input can be cast to an integer
    def validate(self):
        return True

    def userCancelled(self):
        return False

    def get_k(self):
        return int(self.k.get())

class MeansDialog(Dialog):

    def __init__(self, parent, k, means, headers, header_strings):
        # initialize a new field to store the value the user types
        self.k = k
        self.headers = headers
        self.means = means
        self.header_strings = header_strings    
        # call parent constructor
        Dialog.__init__(self, parent)

    def body(self, master):
        tk.Label(master, text="Cluster").grid(row=0, column=0)

        for i in range(self.k):
            tk.Label(master, text="Cluster " + str(i)).grid(row=i+1, column=0)
            for j in range(len(self.headers)):
                tk.Label(master, text=str(self.means[i,j])[:5]).grid(row=i+1, column=j+1)

        for i in range((len(self.headers))):
            tk.Label(master, text=self.header_strings[i]).grid(row=0, column=i+1)

        return   # initial focus

    def apply(self):
        pass
        
    # make sure that the input can be cast to an integer
    def validate(self):
        return True

    def userCancelled(self):
        return False

if __name__ == "__main__":
    dapp = DisplayApp(1200, 1200)
    dapp.main()

