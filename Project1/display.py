# Skeleton Tk interface example
# Written by Bruce Maxwell
# Modified by Stephanie Taylor
# Updated for python 3
#
# Used macports to install
#  python36
#  py36-numpy
#  py36-readline
#  py36-tkinter
#
# CS 251
# Spring 2018

import tkinter as tk
import math
import random

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
        self.root.title("A better name")

        # set the maximum size of the window for resizing
        self.root.maxsize( 1600, 900 )

        # setup the menus
        self.buildMenus()

        # build the controls
        self.buildControls()

        # build the Canvas
        self.buildCanvas()

        # bring the window to the front
        self.root.lift()

        # - do idle events here to get actual canvas size
        self.root.update_idletasks()

        # now we can ask the size of the canvas
        print(self.canvas.winfo_geometry())

        # set up the key bindings
        self.setBindings()

        # set up the application state
        self.objects = [] # list of data objects that will be drawn in the canvas
        self.data = None # will hold the raw data someday.
        self.baseClick = None # used to keep track of mouse movement

        self.pointStack = []


    def buildMenus(self):
        
        # create a new menu
        menu = tk.Menu(self.root)

        # set the root menu to our new menu
        self.root.config(menu = menu)

        # create a variable to hold the individual menus
        menulist = []

        # create a file menu
        filemenu = tk.Menu( menu )
        menu.add_cascade( label = "File", menu = filemenu )
        menulist.append(filemenu)

        # create another menu for kicks
        cmdmenu = tk.Menu( menu )
        menu.add_cascade( label = "Commands", menu = cmdmenu )
        menulist.append(cmdmenu)

        # menu text for the elements
        # the first sublist is the set of items for the file menu
        # the second sublist is the set of items for the option menu
        
        # I removed a lot of junk from the menu text
        # Extension 2 - addition of undo to menu
        menutext = [ ['Quit  Ctl-Q' ],
                     [ 'Clear Data  Ctr-N', 'Undo Last Data Creation Ctr-Z']]

        # menu callback functions (note that some are left blank,
        # so that you can add functions there if you want).
        # the first sublist is the set of callback functions for the file menu
        # the second sublist is the set of callback functions for the option menu
        # Extension 2 - addition of undo to menu
        menucmd = [ [self.handleQuit],
                    [self.clearData, self.undo]]
        
        # build the menu elements and callbacks
        for i in range( len( menulist ) ):
            for j in range( len( menutext[i]) ):
                if menutext[i][j] != '-':
                    menulist[i].add_command( label = menutext[i][j], command=menucmd[i][j] )
                else:
                    menulist[i].add_separator()

    # create the canvas object
    def buildCanvas(self):
        self.canvas = tk.Canvas( self.root, width=self.initDx, height=self.initDy )
        self.canvas.pack( expand=tk.YES, fill=tk.BOTH )
        return

    # build a frame and put controls in it
    def buildControls(self):

        ### Control ###
        # make a control frame on the right
        rightcntlframe = tk.Frame(self.root)
        rightcntlframe.pack(side=tk.RIGHT, padx=2, pady=2, fill=tk.Y)

        # make a separator frame
        sep = tk.Frame( self.root, height=self.initDy, width=2, bd=1, relief=tk.SUNKEN )
        sep.pack( side=tk.RIGHT, padx = 2, pady = 2, fill=tk.Y)

        # use a label to set the size of the right panel
        label = tk.Label( rightcntlframe, text="Control Panel", width=20 )
        label.pack( side=tk.TOP, pady=10 )

        # make a menubutton
        self.colorOption = tk.StringVar( self.root )
        self.colorOption.set("black")
        colorMenu = tk.OptionMenu( rightcntlframe, self.colorOption, 
                                        "black", "blue", "red", "green" ) # can add a command to the menu
        colorMenu.pack(side=tk.TOP)

        # make a button in the frame
        # and tell it to call the handleButton method when it is pressed.
        button = tk.Button( rightcntlframe, text="Update Color", 
                               command=self.handleButton1 )
        button.pack(side=tk.TOP)  # default side is top

        #make a buton in the frame and tell it to call createRandomDataPoints
        randPoints = tk.Button(rightcntlframe, text="Create Random Data Points",
                                command=self.createRandomDataPoints)
        randPoints.pack(side=tk.TOP)

        # create a listbox
        # make it an element of self so that we can access it in other methods
        self.listbox = tk.Listbox(rightcntlframe, height= 4, exportselection=0)
        self.listbox.pack(side=tk.TOP)
        
        #create the options in the listbox
        self.listbox.insert(tk.END, "Random")
        self.listbox.insert(tk.END, "Gaussian")
        self.listbox.insert(tk.END, "X-Gaussian, Y-Random")
        self.listbox.insert(tk.END, "X-Random, Y-Gaussian")
        self.listbox.insert(tk.END, "Make a line")
        self.listbox.insert(tk.END, "Make a dotted line")
        #set the first option to be selected
        self.listbox.selection_set(0)


        return


    #clears anything from self.object and deletes anything on the canvas
    def clearData(self, event=None):
        for thing in self.objects:
            self.canvas.delete(thing)
        self.objects = []    

    #Extension 1
    #Deletes the last data that was added to the canvas
    def undo(self, event=None):
        print(self.pointStack)
        if self.pointStack != []:
            for thing in self.objects[len(self.objects)-self.pointStack[-1]:]:
                self.canvas.delete(thing)

        self.objects = self.objects[:-self.pointStack[-1]]
        self.pointStack = self.pointStack[:-1]

    #Creates 100 data points at random locations with a radius of 2
    def createRandomDataPoints(self, event=None):
        selection = self.listbox.curselection()
        
        # it will be longer than 3 if there is anything in it
        if len(selection) > 0:
            if selection[0] == 0:
                print('Random distribution')
            elif selection[0] == 1:
                print('Gaussian distribution')
            # elif selection


            #the radius of the data points
            dx = 3
            
            if selection[0] < 4:
                #make a new ModalDialog object
                md = ModalDialog(self.root)

                #get the integer from the user
                num = int(md.getVal())

            
            #Extensions 5 and 6 to create lines
            #use ternary operator to determine solidity or dottedness of line
            else:
                ld = LineDialog(self.root)
                slope, intercept = ld.getVal()
                num = int(self.canvas.winfo_width() / (10 if selection[0] == 5 else 1))

            #put the number of points created in the stack of points
            self.pointStack.append(num)
            #make however many data points the user wants us to
            for i in range(num):
                #assign to x and y different random numbers within the scope of the canvas
                # use random distribution if the user selected random 
                if selection[0] == 0:
                    x = random.randint(0,self.canvas.winfo_width())
                    y = random.randint(0,self.canvas.winfo_height())
                
                #use Gaussian distribution if he user selected Gaussian
                elif selection[0] == 1:
                    x = random.gauss(self.canvas.winfo_width()/2, self.canvas.winfo_width()/6)
                    y = random.gauss(self.canvas.winfo_height()/2, self.canvas.winfo_height()/6)
                
                #Extension 3 - Gaussian x and random y to create a vertical bar
                elif selection[0] == 2:
                    x = random.gauss(self.canvas.winfo_width()/2, self.canvas.winfo_width()/24)
                    y = random.randint(0,self.canvas.winfo_height())
                
                #Extension 4 - Random x and Gaussian y to create a horizontal bar
                elif selection[0] == 3:
                    x = random.randint(0,self.canvas.winfo_width())
                    y = random.gauss(self.canvas.winfo_height()/2, self.canvas.winfo_height()/24)
                
                #Extensions 5 and 6
                # 5 creates a solid line
                # 6 creates a dotted line using ternary operator
                elif selection[0] == 4 or selection[0] == 5:
                    print('before')
                    x = i * (10 if selection[0] == 5 else 1)
                    print('after')
                    y = -1 * slope * x  - intercept + self.canvas.winfo_height()/2
                    print(x, y)
                
                #draw the new point     
                pt = self.canvas.create_oval( x-dx, y-dx, x+dx, y+dx,
                            fill=self.colorOption.get(), outline='' )
                #add the new point to self.objects
                self.objects.append(pt)

    def setBindings(self):
        # bind mouse motions to the canvas
        self.canvas.bind( '<Button-1>', self.handleMouseButton1 )
        self.canvas.bind( '<Control-Button-1>', self.handleMouseButton2 )
        self.canvas.bind( '<Shift-Button-1>', self.handleMouseButton3 )
        self.canvas.bind( '<B1-Motion>', self.handleMouseButton1Motion )
        self.canvas.bind( '<Control-B1-Motion>', self.handleMouseButton2Motion )
        self.canvas.bind( '<Shift-B1-Motion>', self.handleMouseButton3Motion )
        # bind command sequences to the root window
        self.root.bind( '<Control-q>', self.handleQuit )
        self.root.bind('<Control-n>', self.clearData)
        self.root.bind('<Control-z>', self.undo)

    def handleQuit(self, event=None):
        print( 'Terminating')
        self.root.destroy()

    def handleButton1(self):
        print( 'handling command button:', self.colorOption.get())
        for thing in self.objects:
            self.canvas.itemconfig(thing, fill=self.colorOption.get())

    def handleMenuCmd1(self):
        print( 'handling menu command 1')

    def handleMouseButton1(self, event):
        print( 'handle mouse button 1: %d %d' % (event.x, event.y))
        self.baseClick = (event.x, event.y)

    def handleMouseButton2(self, event):
        #get the mouse location
        self.baseClick = (event.x, event.y)
        print( 'handle mouse button 2: %d %d' % (event.x, event.y))
        dx = 3
        #set a random color
        rgb = "#%02x%02x%02x" % (random.randint(0, 255), 
                             random.randint(0, 255), 
                             random.randint(0, 255) )
        #make the oval at the location with the random color fill
        oval = self.canvas.create_oval( event.x - dx,
                                        event.y - dx, 
                                        event.x + dx, 
                                        event.y + dx,
                                        fill = rgb,
                                        outline='')
        #put it into self.objects
        self.objects.append( oval )

    def handleMouseButton3(self, event):
        self.baseClick = (event.x, event.y)
        print( 'handle mouse button 3: %d %d' % (event.x, event.y))

    # This is called if the first mouse button is being moved
    def handleMouseButton1Motion(self, event):
        # calculate the difference
        diff = ( event.x - self.baseClick[0], event.y - self.baseClick[1] )
        # update base click
        self.baseClick = ( event.x, event.y )
        print( 'handle button1 motion %d %d' % (diff[0], diff[1]))
        
        for obj in self.objects:
            #get the initial location
            loc = self.canvas.coords(obj)
            #set the new location based on how far the mouse has moved
            self.canvas.coords(
                obj,
                loc[0] + diff[0],
                loc[1] + diff[1],
                loc[2] + diff[0],
                loc[3] + diff[1])
            
    # This is called if the second button of a real mouse has been pressed
    # and the mouse is moving. Or if the control key is held down while
    # a person moves their finger on the track pad.
    def handleMouseButton2Motion(self, event):
        print( 'handle button 2 motion %d %d' % (event.x, event.y) )

    def handleMouseButton3Motion(self, event):
        print( 'handle button 3 motion %d %d' % (event.x, event.y) )
        
    def main(self):
        print( 'Entering main loop')
        self.root.mainloop()

class Dialog(tk.Toplevel):
    #I did not write this class, it was provided via a link from the project page
    def __init__(self, parent, title = None):

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

        self.grab_set()

        if not self.initial_focus:
            self.initial_focus = self

        self.protocol("WM_DELETE_WINDOW", self.cancel)

        self.geometry("+%d+%d" % (parent.winfo_rootx()+50,
                                  parent.winfo_rooty()+50))

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
            self.initial_focus.focus_set() # put focus back
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
        pass #override

    def validate(self):
        return 1 # override

    def apply(self):
        pass # override


class ModalDialog(Dialog):
    
    def __init__(self, parent):
        #initialize a new field to store the value the user types
        self.val = tk.StringVar()
        #call parent constructor
        Dialog.__init__(self, parent)
        
    def body(self, master):
        tk.Label(master, text="Number of Data Points:").grid(row=0)
        self.e1 = tk.Entry(master)
        self.e1.grid(row=0, column=1)

        #set the initial value to 5
        self.e1.insert(tk.END, 5)

        #select the text in the box
        self.e1.selection_range(0,tk.END)
        return self.e1 # initial focus

    def apply(self):
        #put the new value in self.val
        self.val = int(self.e1.get())
        
    #make sure that the input can be cast to an integer
    def validate(self):
        try:
            self.val = int(self.e1.get())
        except ValueError:
            print("Could not convert to an integer.")
            return False
        if self.val > 1000 or self.val < 0:
            print("Please put a value between 0 and 1000")
            return False
        return True

    #getter for self.val
    def getVal(self):
        return self.val

    def userCancelled(self):
        # called by the cancel method, which only the user can 
        # invoke, so anytime we're in here, the user has cancelled
        # so all we have to do is return true
        return True



class LineDialog(Dialog):
    
    def __init__(self, parent):
        #initialize a new field to store the value the user types
        self.slope = tk.StringVar()
        self.intercept = tk.StringVar()
        #call parent constructor
        Dialog.__init__(self, parent)
        
    def body(self, master):
        tk.Label(master, text="Slope:").grid(row=0)
        tk.Label(master, text="Intercept:").grid(row=1)
        
        self.e1 = tk.Entry(master)
        self.e1.grid(row=0, column=1)
        
        self.e2 = tk.Entry(master)
        self.e2.grid(row=1, column=1)
        
        #set the initial value to 5
        self.e1.insert(tk.END, 1)
        self.e2.insert(tk.END, 0)
        #select the text in the box
        self.e1.selection_range(0,tk.END)
        self.e2.selection_range(0,tk.END)
        
        return self.e1 # initial focus

    def apply(self):
        #put the new value in self.val
        self.slope = int(self.e1.get())
        self.intercept = int(self.e2.get())

    #make sure that the input can be cast to an integer
    def validate(self):
        mult = 1
        if self.e1.get()[0] == '-':
            mult = -1
        
        try:
            if mult == 1:
                self.slope = int(self.e1.get())
            else:
                self.slope = int(self.e1.get()[1:]) * -1
        except ValueError:
            print("Could not convert slope to an integer.")
            return False
        
        try:
            self.intercept = int(self.e2.get())
        except ValueError:
            print("Could not convert intercept to an integer.")
            return False
        
        return True

    #getter for self.val
    def getVal(self):
        return self.slope, self.intercept

    def userCancelled(self):
        # called by the cancel method, which only the user can 
        # invoke, so anytime we're in here, the user has cancelled
        # so all we have to do is return true
        return True

if __name__ == "__main__":
    dapp = DisplayApp(1200, 675)
    dapp.main()