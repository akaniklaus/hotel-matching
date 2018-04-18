import pymongo
from urllib import urlretrieve
import eventlet
from eventlet.green import urllib

from settings import MONGODB_SERVER, MONGODB_PORT, IMAGES_PATH
from utils import create_dir_if_needed, load_matches
from os.path import exists, join
from os import makedirs, remove
from shutil import rmtree

from PIL import Image
from time import sleep
import json

MONGODB_SERVER = "52.233.173.236"
MONGODB_PORT = 27017
MONGODB_USER = 'livingrooms:TXrW1IhGzp'

def get_conn_bkg(db_name='booking'):
    """
    Return mongo connection uri.
    """

    uri = 'mongodb://{user_pass}@{server}:{port}/{db}'.format(
        user_pass=MONGODB_USER,
        server=MONGODB_SERVER,
        port=MONGODB_PORT,
        db=db_name
    )
    connection = pymongo.MongoClient(uri)

    return connection

connection = get_conn_bkg('booking')

def save_img(uf):
    url, fpath = uf
    success = False
    attempts = 2
    while attempts and not success:
        attempts -= 1
        try:
            urllib.urlretrieve(url, fpath)
            # check that a valid image was downloaded
            # by trying to open it with PIL
            im = Image.open(fpath)
            success = True
        except Exception as e:
            # remove possible corrupted file
            if exists(fpath):
                remove(fpath)
            # sleep(1)
    if success:
        return url, True
    else:
        return url, False

def fetch_room_images(mongoc, bkgid, update=False):
    # download images
    h = mongoc.find_one({ 'hotel_id': str(bkgid) })
    ufs = []

    if not h:
        return ufs

    # create folder for hotel
    hotel_path = join(IMAGES_PATH, 'booking/%s' % str(bkgid))
    folder_existed = exists(hotel_path)
    if folder_existed:
        if update:
            rmtree(hotel_path)
            makedirs(hotel_path)
    else:
        makedirs(hotel_path)

    if update or not folder_existed:
        for i, r in enumerate(h['rooms']):
            if 'gallery' not in r or not r['gallery']:
                continue
            rname = r['name'].encode('ascii', 'ignore')
            rname = rname.replace('/', ' ')
            room_path = join(hotel_path, 'r%02d - %s' % (i, rname))
            create_dir_if_needed(room_path)
            for u in r['gallery']:
                u = u.replace('square60','max1024x768')
                fname = u.split('/')[-1]
                fpath = join(room_path, fname)
                ufs.append((u, fpath))

    return ufs

if __name__ == '__main__':
    # Images for all matches
    matches = load_matches()
    bkgids = matches.values()

    # Images for eval dataset
    # from build_eval_dataset import bkgids

    # bkgids = bkgids[:30]
    # bkgids = ['177267', '93670', '1395071', '353532', '41403', '92328', '36486', '49930', '77773', '798664', '18469', '404396']
    # bkgids = []

    ufs = []
    for hid in bkgids:
        ufs += fetch_room_images(CONN_BKG, hid, update=True)

    print "Starting to fetch %d images for %d hotels" % (len(ufs), len(bkgids))

    pool = eventlet.GreenPool()
    fails = []
    hits = []
    for u, succeeded in pool.imap(save_img, ufs):
        if succeeded:
            print "got img from %s" % u
            hits.append(u)
        else:
            print "failed to get img for %s" % u
            fails.append(u)

    with open("bkg_img_hits.json", 'w') as f:
        json.dump(hits, f)

    with open("bkg_img_fails.json", 'w') as f:
        json.dump(fails, f)

    print "Fetched %.0f%% of images" % (len(hits) * 100.0 / len(ufs))
