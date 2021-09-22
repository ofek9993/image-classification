from tkinter.ttk import Progressbar
import math
import torchvision
import torch
import random
import numpy as np
from torchvision import transforms as T
import matplotlib.pyplot as plt
import cv2
import os
import shutil
from tkinter import *
from tkinter import messagebox
import tkinter.filedialog as fd
import time


# Loading our model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Need to put the model in evaluation mode

# All the items our model can recognize
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic-light', 'fire-hydrant', 'N/A', 'stop-sign',
    'parking-meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports-ball',
    'kite', 'baseball-bat', 'baseball-glove', 'skateboard', 'surfboard', 'tennis-racket',
    'bottle', 'N/A', 'wine-glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted-plant', 'bed', 'N/A', 'dining-table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell-phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy-bear', 'hair-drier', 'toothbrush', 'hair-brush'
]

# ----- identifiers to later help us classify photos ------

event_identifiers = ('person', 'tie', 'wine-glass', 'cup')

sport_identifiers = ('person','frisbee','kite',)

food_identifiers = ('banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                    'fork', 'knife', 'spoon', 'bowl')

vehicle_identifiers = ('car', 'motorcycle', 'bicycle', 'truck', 'airplane','train')

house_identifiers = ('toilet', 'bed', 'sink', 'refrigerator', 'toothbrush', 'toaster', 'microwave', 'oven',
                     'scissors', 'hair-drier', 'hair-brush', 'remote')

sea_identifiers = ("person",'surfboard', 'boat')
# user pet!!


street_photography_identifiers = ('person', 'traffic-light', 'stop-sign', 'bicycle', 'motorcycle'
                                  'bus',  'umbrella', 'suitcase')

animals = ('horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'bird')

# Identifier dict is built in this fashion { tuple of identifiers : name of the folder we want }, this way we can
# Iterate over each tuple and intersect it
identifier_dict = {event_identifiers: 'Event Photography', sport_identifiers: 'Sport Photography',
                   food_identifiers: 'Food Photography', vehicle_identifiers: 'Vehicle Photography',
                   house_identifiers: 'In-Door Photography',street_photography_identifiers: 'Street Photography',
                   animals: 'Animal Photography'}

# Special sets is our 'sorter', we first check if our image contains a special set listed below and sort by it.
special_sets_dict= {('dog',): 'Pet Photography', ('cat',): 'Pet Photography',("person",'surfboard'):"Sea Photography",
                    ("boat",):"Sea Photography", ( 'baseball-glove',):'Sport Photography',('skis',):'Sport Photography',
                    ('snowboard',):'Sport Photography', ( 'sports-ball',):'Sport Photography',('baseball-bat',): 'Sport Photography',
                    ( 'tennis-racket',):'Sport Photography', ("skateboard","person"):"Sport Photography",
                    ("truck",'traffic-light'):"Street Photography",("truck",'person'):"Street Photography", ("truck",'fire-hydrant'):"Street Photography",
                    ("truck","stop-sign"):'Street Photography', ('person','elephant'):'Safari photos', ('person','zebra'):'Safari photos',
                    ('person','giraffe'):'Safari photos', ('person','elephant','giraffe','zebra'):'Safari photos',
                    ("traffic-light",'car'):"Street Photography",('person','bus'):'Street Photography',
                    ('elephant', 'zebra', 'giraffe') : 'Safari photos', ('person','traffic-light') : 'Street Photography',
                    ('person','fire-hydrant'): 'Street Photography', ('person','stop-sign') : 'Street Photography',
                    ('person', 'umbrella'):'Street Photography', ('traffic-light', 'fire-hydrant'):'Street Photography',
                    ('traffic-light', 'stop-sign'):'Street Photography', ('fire-hydrant', 'stop-sign'):'Street Photography',
                    ('traffic-light', 'fire-hydrant', 'stop-sign'):'Street Photography', ('car','fire-hydrant'):'Street Photography',
                    ('bench','bird'):"park photos", ("truck","fire-hydrant"):'Street Photography', ('motorcycle','fire-hydrant') : 'Street Photography',
                    ("bench",'person'):'park photos', ("bench",'person','bird'):'park photos',('fire-hydrant',):'Street Photography',
                    ('person','tie'):'Event Photography', ('person','wine-glass'):'Event Photography',
                    ('cake','cup'):'Event Photography',('wine-glass',):'Event Photography',
                    ('tie',):'Event Photography',('cake','wine-glass'):'Event Photography',('cake','person'):'Event Photography'}
# ----- end of identifiers list



# An empty list used later to show unrecognized photos
unidentified_images = []


# This function gets an image and makes a prediction, returns the class of the image and list of probabilities
def get_prediction(img, threshold=0.8):
    try:
        # the photo are in numpy array form and for pytorch to read it it needs to be in tensor form
        transform = T.Compose([T.ToTensor()])
        temp_image = transform(img)
        # then we make a prediction
        pred = model([temp_image])  # We have to pass in a list of images
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]  # Bounding Boxes
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        pred_box = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]
        return pred_box, pred_class, pred_score
    except IndexError:
        return 1
        # If we cannot find a prediction then we return a recursion call with a lower threshold


# object detection uses an array of images
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
            boxes, pred_class, pred_score = get_prediction(img, threshold=threshold)
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

        # call a function with image path and a list of identify objects that were in our photos
        # example : [dog,dog,dog,person,person,cat]
        # and then return an organized dictionary as followed : {dog:3,person:2,cat:1}
        instance_dict(images[photos_that_were_classified], pred_class, output_dir)
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


def instance_dict(image_path, pred_class, output_dir='.'):
    # this function outputs a dictionary with the number of instances of each object in the image
    # used for image classification later

    # creating an empty dict
    output_dict = {}

    # pred_class is a list of items that where identified in our photos so we iterate that list and count how many
    # times each item got detect on our photos and then append it to our empty dict
    for i in range(len(pred_class)):
        if pred_class[i] not in output_dict.keys():
            output_dict[pred_class[i]] = 1
        else:
            output_dict[pred_class[i]] += 1

    # then we call our image classification function with our new dict data and image path
    image_classification(output_dict, image_path, output_dir)


# this function create folders. check with a given name if the folder exists if so it move the photo file to that folder
# if not it will make one and then move the file
def create_folder(name, image):
    if os.path.exists(name):
        path = os.path.abspath(name)
        shutil.move(image, path)
    else:
        os.makedirs(name)
        path = os.path.abspath(name)
        shutil.move(image, path)


# ------------------------------------------------------------------------------------------


# returns a list of image paths in a given folder
def load_images(path):
    return [os.path.join(path, f) for f in os.listdir(path) if
            (f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png") or f.endswith('.JPG') or f.endswith('.JPEG')
             or f.endswith('.PNG'))]


# this is the function which will sort photos
def image_classification(count_dict, image_path, output_dir='.'):
    # comparison_count_set takes our count_dict.keys() and converts it into a set. [example: count_dict = {'dog':2,'cat':1}
    # count_dict.keys() = ['dog', 'cat'], comparison_count_set = {'dog', 'cat'}]. This is done in order to intersect sets.
    comparison_count_set = set(count_dict.keys())
    # ----- first trying to classify by spacial cases by the spacial sets that are at the beginning of the code
    # for example (person,bench,bird) would be classified to park photos
    for item in special_sets_dict.keys():
        # checking if the intersection between our set and the special sets is equal to the special set
        # if so then we found a special set in our photo and we classify it as follows
        if len((set(item) & comparison_count_set)) == len(item):
            create_folder(output_dir + '\\' + special_sets_dict[item], image_path)
            return
    # ------- here we split in to spacial cases and make use of our identifiers lists that we made in the start of the code---


    # In this block we sort by the max intersection between the objects in our photos and the identifier dict we
    # set in the beginning
    # temp variables used in order to get the max intersection, set at 0 and []
    temp = 0
    temp_list = []
    # aux_list is a list of all the sets in our photo, used later
    aux_list = []
    # Identifier dict is built in this fashion { tuple of identifiers : name }
    # here, we iterate over the tuples in that dict and intersect it with our photo.
    for list in identifier_dict.keys():
        # intersection len is the power of the intersection group between what we detected in our photo and the identifier set
        intersection_len = len(set(list) & comparison_count_set)
        # this if block checks if our intersection group is greater than temp, if it is, temp is set to be intersection_len
        # and temp_list into list
        if intersection_len > temp:
            temp = intersection_len
            temp_list = list
        # used in order to append all sets into aux_list
        if len((set(list) & comparison_count_set)) > 0:
            aux_list.append(set(list) & comparison_count_set)

    flag = SetlistIntoList(aux_list)
    # if flag == 1, there's no more than one set\group with max power
    if flag == 1 and aux_list!=[]:
        for list in identifier_dict.keys():
            if temp_list == list:
                create_folder(output_dir + '\\'+ identifier_dict[list], image_path)
                return
    else:
        # If we havent found any other category to sort our image by, we check if there is a person in the photo and sort
        # to portraits
        if len({'person'} & comparison_count_set) == 1:
            create_folder(output_dir + '\\Portraits', image_path)
            return
        else:
            unidentified_images.append(image_path)
            return


# -----end of classification

def SetlistIntoList(list):
    # This function receives a list of sets\groups (for example: [{},{'person','cat},{'dog','cat'}],
    # the function starts by checking if there are more than one group of equal power
    # (in our example above we have 2 groups with the power of 2), if that case happens, we sort by the instance we find
    # in our max_dict, if not - we change flag to 1 and continue sorting normally.

    flag = 0
    # power list is a list of integers that represents each set's power (in our example: [0,2,2] )
    power_list = [len(x) for x in list]
    # aux_list is  used to check if we have more than one group of equal power
    aux_list = []
    # str_val is used to extract the strings from our sets and form them into a list
    str_val = ''
    # we iterate over power_list and append to the aux list the groups that have max power
    for i in power_list:
        if i == max(power_list):
            aux_list.append(i)
    # This if statement checks if we have more than one max group
    if len(aux_list) > 1:
        return 0
    else:
        return 1

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
    plt.show(block=False)
    #plt.pause(4)
    #plt.close()
    # --- end block
    # then call create folder to make a folder called unrecognized
    for image in unidentified_images:
        create_folder(output_dir + "\\" + 'Unrecognized', image)


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
            boxes, pred_class, pred_score = get_prediction(img, threshold=threshold)
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



# front end functions
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
# end of gui