import tkinter as tk
root = tk.Tk()
root.title('Blue Machine')

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
				   'Android Ver.: ' + AndroiVerVar.get() + '\n')



def createDropDownMenu(frameLabel, var):
	frame = tk.Frame(root)
	frame.pack(side='left')
	label = tk.Label(frame, text=frameLabel)
	label.pack(side='top')
	#var = tk.StringVar()
	var.set('----') # default value
	var.trace('w', printSelection)
	window = tk.OptionMenu(frame, var, *OPTIONS)
	window.pack(side='bottom')


createDropDownMenu('Category', CategoryVar)
createDropDownMenu('Type', TypeVar)
createDropDownMenu('Content Rating', ContentRatingVar)
createDropDownMenu('Genres', GenresVar)
createDropDownMenu('Android Ver.', AndroiVerVar)

submit = tk.Button(root, text='Submit', command=printSelection)
submit.pack(side='left')
messageFrame = tk.Frame(root)
messageFrame.pack(side='bottom')
selection = tk.Message(messageFrame, textvariable=messageVar)
selection.pack()

root.mainloop()
#popUp.mainloop()