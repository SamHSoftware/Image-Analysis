from tkinter import *
from tkinter import filedialog

def file_selection_dialog():
    root = Tk()
    root.title('Please select the file in question')

    root.filename = filedialog.askopenfilename(initialdir="/", title="Select A File", filetypes=[("All files", "*.*")])
    file_path = root.filename

    root.destroy()
    return file_path