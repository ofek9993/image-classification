import os
import shutil

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

# ----- identifiers to later help us classify photos if our spacial case sorter did not succeed ------

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
# A list to save all of our unrecognized photos
unidentified_images=[]



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

#Our main classification function
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
