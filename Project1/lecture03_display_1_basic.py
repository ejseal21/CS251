'''
lecture03_display_1_basic.py
Demo of Tkinter module GUI features
Oliver W. Layton
CS251: Data Analysis and Visualization

Best site for Tkinter help: http://effbot.org/tkinterbook/
'''

# import tk
import tkinter as tk

class DisplayApp:
    '''Create a class to build and manage the display'''
    def __init__(self, width, height):
        
        # create a tk object, which is the root window
        self.root = tk.Tk()
        

        # width and height of the window
        self.initialWidth = width
        self.initialHeight = height
        

        # set up the geometry for the window. The +values are X and Y offsets.
        # Note: Y is positive going downward
        self.root.geometry(newGeometry=f'{self.initialWidth}x{self.initialHeight}+100+200')

        # set the title of the window
        self.root.title('My app')

        # set the maximum size of the window for resizing
        self.root.minsize(width=300, height=300)
        self.root.maxsize(width=1600, height=1000)

        #make menu bar items
        self.buildMenus()

        # Make the right-hand side control center
        self.buildControls()

        # Make the canvas
        self.buildCanvas()


        # bring the window to the front (vs. lower())
        self.root.lift()

        # - do idle events here to reliably get actual canvas size
        self.root.update_idletasks()

        # now we can ask the size of the canvas
        print(self.root.winfo_geometry())
        pass


    def buildControls(self):
        controlPanel = tk.Frame(master=self.root)
        controlPanel.pack(side=tk.RIGHT, fill=tk.Y)

        sep = tk.Frame(master=self.root, height=self.initialHeight, width=4, borderwidth=3, relief=)
        sep.pack(side=tk.RIGHT)

        label = tk.Label(master=None, text='Control Panel', width=20)
        label.pack(side=tk.TOP)

    def buildCanvas(self):
        self.canvas = tk.Canvas(master=self.root, width=self.initialWidth, height=self.initialHeight)
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)

    def buildMenus(self):
        # Create a root menu object
        menu = tk.Menu(master=self.root)

        # Associate the menu with the app
        self.root.config(menu=menu)

        filemenu = tk.Menu(master=menu)
        menu.add_cascade(label='File', menu=filemenu)

        # Add some actual commands
        filemenu.add_command(label='Quit    Ctrl+Q', command=self.handleQuit)
        filemenu.add_command(label='Clear   Ctrl+C', command=self.handleClear)

        #Make another menu called Command
        cmdMenu = tk.Menu(master=menu)
        menu.add_cascade(label='Command', menu=cmdMenu)

        #add some commands
        cmdMenu.add_command(label='Test1', command=self.handleNothing)
        cmdMenu.add_command(label='Test2', command=self.handleNothing)
        cmdMenu.add_separator()
        cmdMenu.add_command(label='Test3', command=self.handleNothing)
        
        testMenu = tk.Menu(master=cmdMenu)
        cmdMenu.add_cascade(label='Test Menu', menu=testMenu)
        testMenu.add_command(label='test4',command=self.handleNothing)


    def setBindings(self):
        self.root.bind('<Control-q>', self.handleQuit)


    def handleNothing(self, event=None):
        print('Doing nothing')

    def handleQuit(self, event=None):
        print('Quitting the app')
        self.root.destroy()

    def handleClear(self, event=None):
        print('Clearing the data')
        pass




    def main(self):
        print('Entering main loop')
        
        #x1, y1, x2, y2
        self.myLine = self.canvas.create_line(100,100,300,300)
        
        # Run main loop to listen for events
        self.root.mainloop()

if __name__ == "__main__":
    dapp = DisplayApp(1200, 675)
    dapp.main()