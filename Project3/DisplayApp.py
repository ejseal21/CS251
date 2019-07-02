# Skeleton File for CS 251 Project 3
# You can use this file or your file from Project 1
# Delete unnecessary code before handing in

import tkinter as tk
import math
import random
import numpy as np
import View

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
        print(self.endpoints)        
        self.lines = []
        self.buildAxes()
        self.updateAxes()
        # set up the application state  
        self.objects = []
        self.data = None

        #fields to hold the last values of button 1 motion
        self.b1mx = None
        self.b1my = None
        self.vrp = self.view.getVRP()
        self.prev_y = None
        self.base_click2 = []

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
        self.buttons.append( ( 'reset', tk.Button( self.cntlframe, text="Reset", command=self.handleResetButton, width=5 ) ) )
        self.buttons[-1][1].pack(side=tk.TOP)  # default side is top

        return

    # create the axis line objects in their default location
    def buildAxes(self):

        vtm = self.view.build()
        pts = vtm * self.endpoints
        #Extension 1 puts colors on the axes so that the user can tell them apart
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
        self.root.bind( '<Button-1>', self.handleButton1 )
        self.root.bind( '<Button-2>', self.handleButton2 )
        self.root.bind( '<Control-Button-1>', self.handleButton2)
        self.root.bind( '<Button-3>', self.handleButton3 )
        self.root.bind( '<B1-Motion>', self.handleButton1Motion )
        self.root.bind( '<B2-Motion>', self.handleButton2Motion )
        self.root.bind( '<Control-B1-Motion>', self.handleButton2Motion )
        self.root.bind( '<B3-Motion>', self.handleButton3Motion )
        self.root.bind( '<Control-q>', self.handleQuit )
        self.root.bind( '<Control-o>', self.handleModO )
        #Extension 2 lets the user use more intuitive controls to scale
        self.canvas.bind( '<Shift-Button-1>', self.handleButton3)
        self.canvas.bind( '<Shift-B1-Motion>', self.handleButton3Motion)
        return

    def handleOpen(self):
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
        print('handling reset button')

    def handleButton1(self, event):
        print('handle button 1: %d %d' % (event.x, event.y))
        self.b1mx = event.x
        self.b1my = event.y
        self.vrp = self.view.getVRP()

    # rotation
    def handleButton2(self, event):
        self.base_click2.append(event.x)
        self.base_click2.append(event.y)
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
        print(self.vrp)
        self.updateAxes()        
        print('handle button 1 motion: %d %d' % (event.x, event.y) )
    
    def handleButton2Motion(self, event):
        dx = (event.x - self.base_click2[0]) * math.pi/2000
        dy = (event.y - self.base_click2[1]) * math.pi/2000
        self.view = self.view.clone()
        self.view.rotateVRC(dx, dy)
        self.updateAxes()


        print('handle button 2 motion: %d %d' % (event.x, event.y) )

    def handleButton3Motion( self, event):
        #see if this is the first time through this method
        if self.prev_y == None:
            #set previous y value to the initial y value
            self.prev_y = self.button3_location[1]
            scale = 0
        else:
            #scale gets the difference between the prev_y and new y
            scale = self.prev_y - event.y 
            #update prev_y
            self.prev_y = event.y
        
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

        print('handle button 3 motion: %d %d' % (event.x, event.y) )

    def main(self):
        print('Entering main loop')
        self.root.mainloop()

if __name__ == "__main__":
    dapp = DisplayApp(700, 500)
    dapp.main()

