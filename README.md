# ImageClassifier
This project was created jointly by me and a fellow friend in my Electrical Engineering studies Erez Verdnikov (https://github.com/erezberezz)
This classifer uses a neural network trained on the COCO dataset in order to sort photos into common photography generes.

# Libraeris used: 
Torchvision, Torch, Math, Random, Numpy, Matplotlib, cv2, os, Shutil, Tkinter, Time

Python version 3.8

# How to use our program:


We first need to select an input folder that contains the photos we wish to sort, and an output folder in which we sort our photos to
![Image Classification 1](https://user-images.githubusercontent.com/88950513/134409125-9d82091d-e7db-4133-8b29-cf857c1db216.gif)


While its running we can see our photos being sorted
![Image Classification 2](https://user-images.githubusercontent.com/88950513/134409267-f801c6f5-e594-4e2a-8e1d-0499e0cda3f9.gif)



After all thats done, our program informs us of the unrecognized photos
![Image Classification 3](https://user-images.githubusercontent.com/88950513/134409330-f3d763e4-26a4-490a-92fb-ecbc65cdf626.PNG)


Here we browse our sorted photos
![Image Classification 4](https://user-images.githubusercontent.com/88950513/134409414-a1ffe4e9-dd32-4ede-91a2-e084e484191f.gif)



And the user can also debug photos to see how they were recognized
![Image Classification 5](https://user-images.githubusercontent.com/88950513/134410323-7f00ed4f-1ead-40d0-a166-94ae2cb15184.gif)



# How it works:
Our image classifier recives a folder full of images and sorts them by the objects it detects in them.
We use set intersection for our sorting and we go through 3 steps in order to sort our photos:


First step is to go through a list of "Special" cases in the form of tuples, we sort by it if our special case tuple is a subset of our photo's object set.


After that, we go iterate over a dictionary containing diffrenet types of photo idetifier tuples (i.e. Street Photography identifiers, Sports Photography and etc.) and sort by the biggest intersection between our photo's object set and the identifier tuple.


Finally, if we didnt find any match in the first two steps, if we have people in our photo we sort into Portraits, else we put it into unrecogizned folder
