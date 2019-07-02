# Skeleton File for CS 251 Project 3
# You can use this file or your file from Project 1
# Delete unnecessary code before handing in

import tkinter as tk
import math
import random
import numpy as np
import View
from tkinter import filedialog
import data
import analysis

# create a class to build and manage the display
class DisplayApp:

    def __init__(self, width, height):

        # create a tk object, which is the root window
        self.root = tk.Tk()

        # width and height of the window
        self.initDx = width
        self.initDy = height

        # set up the geometry for the window
        self.root.geometry( "%dx%d+50+30" % (self.initDx, self.initDy) )

        # set the title of the window
        self.root.title("Viewing Axes")

        # set the maximum size of the window for resizing
        self.root.maxsize( 1024, 768 )

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
        self.view = View.View()
        
        # Create the axes fields and build the axes
        self.endpoints = np.matrix([[0,1,0,0,0,0],
                                    [0,0,0,1,0,0],
                                    [0,0,0,0,0,1],
                                    [1,1,1,1,1,1]])
        # print(self.endpoints)        
        self.lines = []
        self.buildAxes()
        # self.updateAxes()
        # self.updatePoints()
        # set up the application state  
        self.objects = []
        self.data = None
        self.points = []
        #fields to hold the last values of button 1 motion
        self.b1mx = None
        self.b1my = None
        self.vrp = self.view.getVRP()
        self.prev_y = None
        self.base_click2 = []
        self.rot_x = None
        self.rot_y = None
        self.headers = None
        #matrix to hold the points to plot
        self.plot = None

        #Flags to determine whether or not the program should plot using some scale
        self.menuFlags = [False, False, False, False, False]

    def buildMenus(self):
        
        # create a new menu
        self.menu = tk.Menu(self.root)

        # set the root menu to our new menu
        self.root.config(menu = self.menu)

        # create a variable to hold the individual menus
        self.menulist = []

        # create a file menu
        filemenu = tk.Menu( self.menu )
        self.menu.add_cascade( label = "File", menu = filemenu )
        self.menulist.append(filemenu)


        # menu text for the elements
        menutext = [ [ 'Open...  \xE2\x8C\x98-O', '-', 'Quit  \xE2\x8C\x98-Q' ] ]

        # menu callback functions
        menucmd = [ [self.handleOpen, None, self.handleQuit]  ]
        
        # build the menu elements and callbacks
        for i in range( len( self.menulist ) ):
            for j in range( len( menutext[i]) ):
                if menutext[i][j] != '-':
                    self.menulist[i].add_command( label = menutext[i][j], command=menucmd[i][j] )
                else:
                    self.menulist[i].add_separator()

    # create the canvas object
    def buildCanvas(self):
        self.canvas = tk.Canvas( self.root, width=self.initDx, height=self.initDy )
        self.canvas.pack( expand=tk.YES, fill=tk.BOTH )
        return

    # build a frame and put controls in it
    def buildControls(self):

        # make a control frame
        self.cntlframe = tk.Frame(self.root)
        self.cntlframe.pack(side=tk.RIGHT, padx=2, pady=2, fill=tk.Y)

        sep = tk.Frame( self.root, height=self.initDy, width=2, bd=1, relief=tk.SUNKEN )
        sep.pack( side=tk.RIGHT, padx = 2, pady = 2, fill=tk.Y)

        # make a cmd 1 button in the frame
        self.buttons = []
        self.buttons.append(('new window', tk.Button(self.cntlframe, text='Make a New Window', command=self.handleNewWindow,width=16)))
        self.buttons[-1][1].pack(side=tk.TOP)  # default side is top       
        self.buttons.append( ( 'reset', tk.Button( self.cntlframe, text="Reset", command=self.handleResetButton, width=16 ) ) )
        self.buttons[-1][1].pack(side=tk.TOP)  # default side is top
        self.buttons.append(('plot data', tk.Button(self.cntlframe, text ="Plot Data", command=self.handlePlotData, width=16)))
        self.buttons[-1][1].pack(side=tk.TOP)  # default side is top

        #We will use this in handleChooseAxes() to 
        self.optionMenus = []
        return

    def handleNewWindow(self):
        newdapp = DisplayApp(700, 500)

    # create the axis line objects in their default location
    def buildAxes(self):

        vtm = self.view.build()
        pts = vtm * self.endpoints
        #Extension 3 puts colors on the axes so that the user can tell them apart
        self.lines.append(self.canvas.create_line(pts[0,0], pts[1,0], pts[0,1], pts[1,1], fill="blue"))
        self.lines.append(self.canvas.create_line(pts[0,2], pts[1,2], pts[0,3], pts[1,3], fill="red"))
        self.lines.append(self.canvas.create_line(pts[0,4], pts[1,4], pts[0,5], pts[1,5], fill="green"))
        
        #Extension 3 labels the axes
        self.xlabel = tk.Label(self.canvas, text = 'x')
        self.xlabel.place(x=pts[0,1], y=pts[1,1])
        self.ylabel = tk.Label(self.canvas, text = 'y')
        self.ylabel.place(x=pts[0,3], y=pts[1,3])
        self.zlabel = tk.Label(self.canvas, text = 'z')
        self.zlabel.place(x=pts[0,5], y=pts[1,5])

    # modify the endpoints of the axes to their new location
    def updateAxes(self):
        vtm = self.view.build()
        pts = vtm * self.endpoints
        for i in range(len(self.lines)):
            self.canvas.coords(self.lines[i],[pts[0,i*2], pts[1,i*2], pts[0,i*2 + 1], pts[1,i * 2 + 1]])
       
         #Extension 3 also updates the locations of the labels as the axes move
        self.xlabel.place(x=pts[0,1], y=pts[1,1])
        self.ylabel.place(x=pts[0,3], y=pts[1,3])
        self.zlabel.place(x=pts[0,5], y=pts[1,5])

    def setBindings(self):
        self.canvas.bind( '<Button-1>', self.handleButton1 )
        self.canvas.bind( '<Button-2>', self.handleButton2 )
        self.canvas.bind( '<Control-Button-1>', self.handleButton2)
        self.canvas.bind( '<Button-3>', self.handleButton3 )
        self.canvas.bind( '<B1-Motion>', self.handleButton1Motion )
        self.canvas.bind( '<B2-Motion>', self.handleButton2Motion )
        self.canvas.bind( '<Control-B1-Motion>', self.handleButton2Motion )
        self.canvas.bind( '<B3-Motion>', self.handleButton3Motion )
        self.root.bind( '<Control-q>', self.handleQuit )
        self.root.bind( '<Control-o>', self.handleModO )
        #Extension 2 lets the user use more intuitive controls to scale
        self.canvas.bind( '<Shift-Button-1>', self.handleButton3)
        self.canvas.bind( '<Shift-B1-Motion>', self.handleButton3Motion)
        return

    def buildPoints(self, headers): 
        #delete existting canvas objects used for plotting data
        for point in self.points:
            self.canvas.delete(point)
        self.points = []
        self.plot_data = analysis.normalize_columns_separately(headers, self.data)
        self.plot = self.plot_data[:,:2]
        print("normal stuff:", self.plot)
        z_flag = False
        if self.menuFlags[2]:
            self.plot = np.hstack((self.plot,self.plot_data[:,2]))
            z_flag = True
        else:
            self.plot = np.hstack((self.plot, np.zeros((len(self.plot),1))))
        
        size_flag = False
        if self.menuFlags[3]:
            size_flag = True
            if z_flag:
                self.size = self.plot_data[:,3]
            else:
                self.size = self.plot_data[:,2]
        else:
            self.size = np.ones((len(self.plot),1))    
        self.size = 3 * self.size + 1
 
        if self.menuFlags[4]:
            if z_flag and size_flag:
                color = self.plot_data[:,4]
            elif (z_flag and not size_flag) or (not z_flag and size_flag):
                color = self.plot_data[:,3]
            else:
                color = self.plot_data[:,2]
            
            self.green = -255 * color + 255
            self.red = 255 * color
        else:
            self.green = np.ones((len(self.plot),1))
            self.red = np.ones((len(self.plot),1))

        #homogeneous coordinate
        self.plot = np.hstack((self.plot, np.ones((self.plot.shape[0],1))))

        #make a vtm so the points aren't tiny
        vtm = self.view.build()

        print("plot:", self.plot)
        #put the points through the vtm
        pts = (vtm * self.plot.T).T
        print("points", pts)
        #loop over the points, drawing each one
        for i in range(len(pts)):
            x = pts[i,0]
            y = pts[i,1]

            #Extension 1 gives the user the capability to use different shapes
            if self.vshape.get() == 'Circle':
                pt = self.canvas.create_oval( int(x-self.size[i]), int(y-self.size[i]), 
                    int(x+self.size[i]), int(y+self.size[i]), fill="#%02x%02x%02x" % (int(self.red[i]), 
                        int(self.green[i]),0), outline='')
            elif self.vshape.get() == 'Rectangle':
                pt = self.canvas.create_rectangle( int(x-self.size[i]), int(y-self.size[i]), 
                    int(x+self.size[i]), int(y+self.size[i]), fill="#%02x%02x%02x" % (int(self.red[i]), 
                        int(self.green[i]),0), outline='')
            elif self.vshape.get() == 'Triangle':
                pt = self.canvas.create_polygon(int(x-self.size[i]), int(y+self.size[i]), int(x+self.size[i]), 
                    int(y+self.size[i]), int(x), int(y-self.size[i]))
            elif self.vshape.get() == 'Pentagon':
                pt = self.canvas.create_polygon(int(x-self.size[i]), int(y), int(x), int(y+self.size[i]), int(x+self.size[i]),
                int(y), int (x+self.size[i]), int(y-self.size[i]), int(x-self.size[i]), int(y-self.size[i]))
            else:
                pt = self.canvas.create_arc( int(x-self.size[i]), int(y-self.size[i]), 
                    int(x+self.size[i]), int(y+self.size[i]), fill="#%02x%02x%02x" % (int(self.red[i]),
                     int(self.green[i]),0), outline='')
            #put the point object into self.points
            self.points.append(pt)
        return

    def updatePoints(self):
        #don't do anything if there aren't any points
        if len(self.points) == 0:
            return
        #make a new vtm and use it to transform the points
        vtm = self.view.build()
        pts = (vtm * self.plot.T).T
        
        #radius of points
        dx = 1
        dy = 1
        #update the coordinates of each point
        for i in range(pts.shape[0]):
            for j in range(pts.shape[1]):
                x = pts[i,0]
                y = pts[i,1]
                if self.vshape.get() == 'Triangle':
                    self.canvas.coords(self.points[i],( int(x-self.size[i]), int(y+self.size[i]), 
                        int(x+self.size[i]), int(y+self.size[i]), int(x), int(y-self.size[i])))
                elif self.vshape.get() == 'Pentagon':
                    pt = self.canvas.coords(self.points[i],(int(x-self.size[i]), int(y), int(x), int(y+self.size[i]),
                     int(x+self.size[i]), int(y), int (x+self.size[i]), int(y-self.size[i]), int(x-self.size[i]), int(y-self.size[i])))
                else:
                    self.canvas.coords(self.points[i], (int(x-self.size[i]), int(y-self.size[i]), 
                int(x+self.size[i]), int(y+self.size[i])))
        print("self.vx", self.vx.get())
        print("self.headers[0]",self.headers[0])
        return

    def handlePlotData(self):
        #The user will hit plot after they have selected which axes to plot (hopefully)
        self.headers = [self.vx.get(), self.vy.get(), self.vz.get(), self.vcolor.get(), self.vsize.get()]
        
       #set the menuFlags to True if the user wants to plot it
        for i in range(5):
            if self.headers[i] != 'None':
                self.menuFlags[i] = True
        print("self.menuFlags",self.menuFlags)
        #get rid of None headers with this loop
        i = 4
        while i >= 0:
            if self.headers[i] == 'None':
                del self.headers[i]
            i-=1

        #put the points on the screen
        self.buildPoints(self.headers)
        
    #returns the first three headers of the data
    def handleChooseAxes(self):
        #put data headers into a list
        self.headers = self.data.get_headers()
        choices = tuple(self.headers)
        print("self.headers1",self.headers)
        
        #we need the StringVar to tell which option is selected
        self.vx = tk.StringVar(self.root)
        #set default selected option to the first item in headers
        self.vx.set(self.headers[0])
        #* dereferences choices so we can pass in an arbitrary number of arguments
        xfeat = tk.OptionMenu(self.cntlframe, self.vx, *choices)
        #put it into the cntlframe
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
        
        shapes = ('Circle', 'Rectangle', 'Triangle', 'Pentagon','Arc')
        self.vshape = tk.StringVar(self.root)
        self.vshape.set('Circle')
        shape_option = tk.OptionMenu(self.cntlframe, self.vshape, *shapes)
        shape_option.pack(side=tk.TOP)

        return self.headers

    def handleOpen(self):
        fn = filedialog.askopenfilename(parent=self.root, 
                    title='Choose a data file', initialdir='.')
        self.data = data.Data()
        self.data.read(fn)
        self.handleChooseAxes()
        print('handleOpen')

    def handleModO(self, event):
        self.handleOpen()

    def handleQuit(self, event=None):
        print('Terminating')
        self.root.destroy()

    #Extension 4 implements reset
    def handleResetButton(self):
        self.view.reset()
        self.updateAxes()
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
        #get the changes in x and y
        if self.b1mx == None:
            self.b1mx = event.x
        if self.b1my == None:
            self.b1my =event.y
        dx = event.x - self.b1mx
        dy = event.y - self.b1my
        
        #divide by screen size
        screen = self.view.getScreen()
        dx /= screen[0]
        dy /= screen[1]

        #multiply by extent
        extent = self.view.getExtent()
        dx *= extent[0,0]
        dy *= extent[0,1]

        #update VRP
        self.view.setVRP( self.vrp + dx * self.view.getU() + dy * self.view.getVUP())
        # print(self.vrp)
        self.updateAxes()   
        self.updatePoints()     
        print('handle button 1 motion: %d %d' % (event.x, event.y) )
    
    #Rotation
    def handleButton2Motion(self, event):
        dx = (self.base_click2[0] - event.x) * math.pi/200
        dy = (self.base_click2[1] - event.y) * math.pi/200
        self.view = self.clone.clone()
        self.view.rotateVRC(dx, dy)
        self.updateAxes()
        self.updatePoints()
        print('handle button 2 motion: %d %d' % (event.x, event.y) )

    #Scaling
    def handleButton3Motion( self, event):
        #see if this is the first time through this method
        #scale gets the difference between the prev_y and new y
        scale = self.button3_location[1] - event.y 
        #update prev_y
        self.button3_location[1] = event.y
        
        #doing this sets the default to 1 and makes it so that moving
        #the mouse a lot doesn't affect it way too much
        scale += 100
        scale /= 100
        
        #keep the scale between 3 and 0.1
        if scale > 3.0:
            scale = 3.0
        elif scale < 0.1:
            scale = 0.1
        
        self.view.setExtent((1/scale)*self.view.getExtent())
        self.updateAxes()
        self.updatePoints()

        print('handle button 3 motion: %d %d' % (event.x, event.y) )

    def main(self):
        print('Entering main loop')
        self.root.mainloop()

if __name__ == "__main__":
    dapp = DisplayApp(700, 500)
    dapp.main()

