# Imports of python packages
import math
import random
import matplotlib.pyplot as plt
import cv2
import os
#--- End

#All of the imports of python files that where writen specificly for this project
from predictions import get_prediction as predict
from classifier import  instance_dict as image_classifier
from classifier import unidentified_images
from classifier import create_folder as create
#----End

#--All of GUI imports
from tkinter.ttk import Progressbar
from tkinter import *
from tkinter import messagebox
import tkinter.filedialog as fd
import time
#------ End

#---------------This is where our Main backend functions work with our other python files------------------------------

# object detection is mostly our main function in this project.
# It uses an array of images and other python files that we have writen to sort and predict images
def object_detection(threshold=0.8):
    # gui popup to get the path of all the photos that we need to classify
    images = load_images(AskForPath())
    # request an output path of were to classify photos
    messagebox.showinfo('Output path ','Please select an output folder')
    output_dir = AskForPath()
    # checks if there are no images at all
    if images == []:
        # if the list is empty then it shows us an error and we can try again
        messagebox.showerror('Error','No images were found!')
        app.mainloop()
    if output_dir == '':
        output_dir = '.'
    # some variables that we will used in our while loop that will helps us with updating our prograss bar
    # and get image path from images list
    photos = len(images)
    photos_that_were_classified = 0
    speed = 1
    while (photos_that_were_classified<photos):
        # hold
        time.sleep(0.05)
        # update bar value
        bar['value'] += (speed / photos) * 100
        # read our photo file as a numpy array
        img = cv2.imread(images[photos_that_were_classified])
        try:
            # try to predict
            boxes, pred_class, pred_score = predict(img, threshold=threshold)
        except TypeError:
            # if the photo was unidentified it then append to that list which later on we will use to plot all the
            # unidentified photos
            unidentified_images.append(images[photos_that_were_classified])
            # update our progress bar
            photos_that_were_classified += speed
            percent.set(str((math.floor((photos_that_were_classified / photos) * 100))) + "%")
            text.set(str(photos_that_were_classified) + "/" + str(photos) + " photos were classified")
            app.update_idletasks()
            # ---end of bar update
            continue

        # call a function with image path and a list of identify objects (pred_class) that were in our photos
        # example : [dog,dog,dog,person,person,cat]
        # And then it classifies images according our sort algorithm
        image_classifier(images[photos_that_were_classified], pred_class, output_dir)
        # -----

        # update our progress bar
        photos_that_were_classified += speed
        percent.set(str((math.floor((photos_that_were_classified / photos) * 100))) + "%")
        text.set(str(photos_that_were_classified) + "/" + str(photos) + " photos were classified")
        app.update_idletasks()
    # then we want the user to be able to see all the photos that weren't recognized
    if unidentified_images == []:
        # keeps the pop up running waiting for user exit command
        app.mainloop()
    else:
        # call the function that plot those images
        error_recognition(unidentified_images,output_dir)


# returns a list of image paths in a given folder
def load_images(path):
    return [os.path.join(path, f) for f in os.listdir(path) if
            (f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png") or f.endswith('.JPG') or f.endswith('.JPEG')
             or f.endswith('.PNG'))]



# this is the function that let us plot all the unrecognized photos
def error_recognition(unidentified_images,output_dir='.'):
    # this block of code i found to be the best size of our plotting canvas
    canvas_size = len(unidentified_images)
    # if the canvas size is even then just / by 2 keep it even
    if canvas_size % 2 == 0:
        if canvas_size == 2:
            canvas_size = canvas_size
        else:
            canvas_size = canvas_size // 2
    else:
        if canvas_size == 1:
            canvas_size = canvas_size
        else:
            canvas_size = (canvas_size // 2) + 1
    # -----end block

    # ----block that plots our photos together
    for i in range(len(unidentified_images)):
        image = cv2.imread(unidentified_images[i])
        plt.subplot(canvas_size,2 , i + 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image)
    plt.suptitle("We couldn't classify these images\n"
                 'They were moved to "Unrecognized" folder')
    #plot images
    plt.show(block=False)
    #4 sec pause for the user to see the plot
    plt.pause(4)
    #then automatically close it
    plt.close()
    # --- end block

    # then call create to make a folder called unrecognized
    for image in unidentified_images:
        create(output_dir + "\\" + 'Unrecognized', image)


# This func allows the user to debug photos and see how they were recognized
def user_view_prediction(threshold=0.7, rect_th=4, text_size=1.5, text_th=2):

    images = AskForPath(flag=1)
    for image in images:
        img = cv2.imread(image)
        text_size, text_th = 1.5, 2
        # here we scale our text size by our image resolution
        scale_factor = max(img.shape[1] // 1000,img.shape[0] // 1000)
        if scale_factor == 0:
            scale_factor = 1
        text_size *= scale_factor
        text_th *=scale_factor
        try:
            boxes, pred_class, pred_score = predict(img, threshold=threshold)
        except TypeError:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # in order to display images with bounding boxes around them
        # I needed to convert a list of float tuples into int tuples
        # ---- and then this block of code plot it on the screen-----
        for i in range(len(boxes)):
            j = 0
            # converting float tuples into int tuples
            while j < 2:
                boxes[i][j] = (int(boxes[i][j][0]), int(boxes[i][j][1]))
                j += 1
            # pred_score_percentages is for displaying the percentage of each prediction
        pred_score_percentages = [round(num * 100, 2) for num in pred_score]
        # this for block is for drawing the bounding boxes around the objects detected
        # can be deleted later (lame) or be made optional for the viewer to see (cooler)
        for i in range(len(boxes)):
            r, g, b = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Random Color
            cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(r, g, b), thickness=rect_th)  # Draw Rectangle with cords
            cv2.putText(img, pred_class[i] + ':' + str(pred_score_percentages[i]) + '%', boxes[i][0],
                        cv2.FONT_HERSHEY_SIMPLEX, text_size, (r, g, b), thickness=text_th)
        plt.figure(figsize=(15, 15))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()
    else:
        return 0
        # ---- end block
# ----------------------------End of Main backend functions -----------------------------------------------

# -------------------------------Start of Gui frontend ------------------------------------
def AskForPath(flag=0):
    # flag = 0 for input\output folder, flag = 1 for selecting multiple files
    if flag == 0:
        path = fd.askdirectory(title='Select Folder')  # shows dialog box and return the path
        path = path.replace('/', '\\')

    elif flag == 1:  # could be just 'else:'
        path = fd.askopenfilenames(title='Choose files')
        # askopenfilenames returns a tuple of strings, this for loop converts it into a string list
        temp = list(path)
        for i in range(len(temp)):
            temp[i] = temp[i].replace('/','\\')
        return temp
    return path

# close function that we use in our exit button
def close():
    app.destroy()
# end of front end functions

#  GUI
app = Tk()
app.title('Image Classifier')
app.geometry('550x250')
# Variables
percent = StringVar()
text = StringVar()
current_value = IntVar()
# ------ end of variables
# accuracy slide bar
slider = Scale(app, from_=1, to=100, orient='horizontal', variable=current_value, label='Accuracy:',length=200)
# default accuracy
slider.set(85)
# all the main menu buttons
b1 = Button(app, text='Select Folder', width=12, command=lambda: object_detection(threshold=current_value.get()/100.0))
b2 = Button(app, text='Debug photos', width=12, command=lambda: user_view_prediction(threshold=current_value.get()/100.0))
bar = Progressbar(app, orient='horizontal', length=300)
percentLabel = Label(app, textvariable=percent)
taskLabel = Label(app, textvariable=text)
# placing
b1.place(height=20, width=100,x=150,y=40)
b2.place(height=20, width=100,x=290,y=40)
slider.place(height=80, width=150, x=200, y=80)
bar.place(height=20, width=200, x=175, y=150)
percentLabel.place(height=10, width=50, x=250, y=200)
taskLabel.place(height=10, width=200, x=175, y=220)
b3 = Button(app, text='Exit', width=12, command=close)
b3.place(height=20,width=100,x=430,y=220)
# UI design
icon = PhotoImage(file=".\\RESOURCES\\Icon.png")
app.iconphoto(False, icon)
app.mainloop()
# --------------------------------------------End Gui -----------------------------------------------------------------
