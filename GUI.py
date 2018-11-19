import Tkinter as tk
from PIL import Image, ImageTk
import os
root = tk.Tk()
root.title('Blue Machine')
root.geometry('600x700')
root.resizable(width = False, height = False)
mainColor = 'misty rose'
root['bg']=mainColor

appPath = os.path.abspath(os.path.dirname(__file__))

topFrame = tk.Frame(root)
topFrame.pack(side='top')
panelImage = ImageTk.PhotoImage(file = os.path.join(appPath, 'panel.png'))
topLabel = tk.Label(topFrame, image=panelImage)
topLabel.pack()

midFrame = tk.Frame(root)
midFrame.pack()

emptyFrame = tk.Frame(root, width=600, heigh=30, bg=mainColor)
emptyFrame.pack()

bottomFrame = tk.Frame(root, bg=mainColor)
bottomFrame.pack()

aBottomFrame = tk.Frame(root)
aBottomFrame.pack()


OPTIONS = ['one', 'two', 'three', 'four', 'five']
CategoryVar = tk.StringVar()
TypeVar = tk.StringVar()
ContentRatingVar = tk.StringVar()
GenresVar = tk.StringVar()
AndroiVerVar = tk.StringVar()
messageVar = tk.StringVar()

def printSelection():
	messageVar.set('Category: ' + CategoryVar.get() + '\n'
				   'Type: ' + TypeVar.get() + '\n'
				   'Content Rating: ' + ContentRatingVar.get() + '\n'
				   'Genres: ' + GenresVar.get() + '\n'
				   'Android Version: ' + AndroiVerVar.get() + '\n')



def createDropDownMenu(frameLabel, var):
	frame = tk.Frame(midFrame, height=100, width=00, bg=mainColor)
	frame.pack(side='left')
	label = tk.Label(frame, text=frameLabel, font='Helvetica 14 bold', bg=mainColor)
	label.pack(side='top')
	var.set('------') # default value
	var.trace('w', printSelection)
	window = tk.OptionMenu(frame, var, *OPTIONS)
	window.pack(side='bottom')


createDropDownMenu('Category', CategoryVar)
createDropDownMenu('Type', TypeVar)
createDropDownMenu('Content Rating', ContentRatingVar)
createDropDownMenu('Genres', GenresVar)
createDropDownMenu('Android Ver.', AndroiVerVar)

submitIcon = ImageTk.PhotoImage(file = os.path.join(appPath, 'submit.png'))
submit = tk.Button(bottomFrame, image=submitIcon, command=printSelection)
submit.pack(side='top')
selection = tk.Message(bottomFrame, textvariable=messageVar, bg=mainColor)
selection.pack(side='bottom')


predictFrame = tk.Frame(aBottomFrame)
predictFrame.pack(side='left')
predictLabel = tk.Label(predictFrame, text='Predict Value', font='Helvetica 14 bold', bg=mainColor)
predictLabel.pack()


root.mainloop()