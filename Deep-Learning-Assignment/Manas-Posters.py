

import matplotlib.pyplot as plt
import scipy.misc as scpm
import pandas as pd
import glob
import numpy as np
import shelve

movie_csv_path = "/Volumes/TSARSHAH 1/Datasets/Manas/Movie-Posters/movie_poster.csv"
posters_path = "/Volumes/TSARSHAH 1/Datasets/Manas/Movie-Posters/posters"

def shelf_open():
    shelf = shelve.open("/Volumes/TSARSHAH 1/Datasets/Manas/Movie-Posters/saved_img_dictionary")

    return shelf

data_ = pd.read_csv(movie_csv_path)

data = data_.drop(['image_url', 'url', 'year'], axis = 1)

image_glob = glob.glob(posters_path + "/" + "*.jpg")
img_dict = {}

def get_id(filename):

    index_s = filename.rfind("/t") + 3
    index_f = filename.rfind('.jpg')

    return filename[index_s:index_f]

def get_images_batches(startIndex, endIndex):

    image_dict = {}

    for i in range(startIndex, endIndex):

        _id = get_id(image_glob[i])
        #print(_id)

        if startIndex >= len(image_glob) or endIndex >= len(image_glob):
            print("Index out of range")
            return None

        try:
            image_dict[_id] = scpm.imread(image_glob[i])

        except:
            print("Unable to read images")
            pass



    return image_dict


def get_img_dict():

    shelf = shelf_open()

    j = 0

    for i in range(0, len(image_glob), 25):

        dict_ = get_images_batches(i, i + 25)

        name = 'img' + str(j) + "_dict"
        print("Name is {}".format(name))

        j += 1

        shelf[name] = dict_

    print('DONE')

    shelf.close()

get_img_dict()
