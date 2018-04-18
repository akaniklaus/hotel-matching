from os import listdir
from os.path import join
import numpy as np
from random import sample
from settings import IMAGES_PATH

# Fetch hotels with many rooms (at least 4)
# and many images per room (at least 8)

# Randomly separate half of the images of each room as queries (Amadeus)
# and keep the rest of them as database (Booking)

def build_eval_dataset():
    queries = {}
    database = {}

    imgcounts = {}
    roomcounts = {}
    for hotel in listdir(IMAGES_PATH):
        hpath = join(IMAGES_PATH, hotel)
        rooms = listdir(hpath)
        if len(rooms) >= 4:
            imgs = {}
            for room in rooms:
                imgs[room] = listdir(join(hpath, room))
            if all([len(l) >= 8 for l in imgs.values()]):
                queries[hotel] = {}
                database[hotel] = {}
                # split room images in queries and database
                for room, imgl in imgs.items():
                    queries[hotel][room] = []
                    database[hotel][room] = []                
                    n = len(imgl)
                    qinds = sample(range(n), n//2)
                    for i, img in enumerate(imgl):
                        if i in qinds:
                            queries[hotel][room].append(img)
                        else:
                            database[hotel][room].append(img)

    return database, queries

database, queries = build_eval_dataset()
bkgids = list(database.keys())