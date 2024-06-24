import tkinter as tk
from tkinter import filedialog
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
from tkinter import *

win = tk.Tk()
lbl = Label (win, text ="", fg = 'black')
lbl.pack()

def bl_click():
    global path2
    try:
       json_file = open ('model1.json', 'r')
       loaded_model_json = json_file.read()
       json_file.close()
       loaded_model = model_from_json (loaded_model_json)
     # load weights into new model
       loaded_model.load_weights("modell.h5")

       label = ["Apple_black_rot", "Apple_leaf_healthy", "Apple_scab_leaf"]

       path2 = filedialog.askopenfilename()
       print (path2)

       test_image = image.load_img (path2, target_size=(128,128))
       test_image = image.img_to_array (test_image)
       test_image = np.expand_dims (test_image, axis = 0)
       result = loaded_model.predict (test_image)
       print (result)
       labe12 = label [result.argmax ()]
       print (label2)
       lbl.configure (text = label2)
       win.mainloop()
    except IOError:
       pass
label1=Label (win, text="GUI For Leaf Disease Detection using OpenCV", fg='blue')
label1.pack()
b1 = tk.Button (win, text="browse image", width=25,height=3, fg='red', command = bl_click)
b1.pack()
win.geometry("550x250")
win.title("Leaf Disease Detection using OpenCV")
win.bind("<Return>", bl_click)
win.mainloop()
