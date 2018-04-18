import pymongo
import eventlet
from eventlet.green import urllib

from settings import MONGODB_SERVER, MONGODB_PORT, IMAGES_PATH
from utils import create_dir_if_needed
from os.path import exists, join
from os import makedirs, remove
from shutil import rmtree

from PIL import Image
from time import sleep
import psycopg2
import json

def get_conn_amd():
    c = psycopg2.connect(database='d36tike0jg2ieb',
                        user='u4esungn748mp7',
                        password='pb31e2aedb696ceb6dffc2c4f0756fd59ec9838854b739fe027d7cc44a2c13fcb',
                        host='ec2-52-212-211-39.eu-west-1.compute.amazonaws.com',
                        port=5432)
    return c
CONN_AMD = get_conn_amd()

def save_img(row):
    prop_id, photo_id, photo_url = row
    # create folder for hotel
    hotel_path = join(IMAGES_PATH, 'amadeus/%s' % str(prop_id))
    if not exists(hotel_path):
        makedirs(hotel_path)

    fpath = join(hotel_path, "%s.jpg" % photo_id)
    photo_url = "http:%s" % photo_url

    success = False
    attempts = 2
    while attempts and not success:
        attempts -= 1
        try:
            urllib.urlretrieve(photo_url, fpath)
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
        return photo_id, photo_url, fpath, True
    else:
        return photo_id, photo_url, fpath, False


def fetch_img_download_data(conn_amd, hids=None):
    cur = conn_amd.cursor()

    query = """SELECT pr.id, ph.id, ph.url
               FROM public."Photos" ph 
               JOIN public."Properties" pr 
                ON ph.relation = pr.id
               JOIN public."VendorIds" v
                ON v.relation = pr.id
               WHERE v.vendor = 'amadeus' AND ph.type = 'Guest room'
            """

    if hids:
        query += ' AND pr.id IN ( %s )' % ','.join("'%s'" % x for x in hids)

    cur.execute(query)
    results = cur.fetchall()
    cur.close()

    return results

if __name__ == '__main__':
    results = fetch_img_download_data(CONN_AMD)

    print "Starting to fetch %d images" % len(results)

    pool = eventlet.GreenPool()
    fails = []
    hits = []
    for i, u, succeeded in pool.imap(save_img, results):
        if succeeded:
            print "%s: got img from %s" % (i, u)
            hits.append(u)
        else:
            print "%s: failed to get img from %s" % (i, u)
            fails.append(u)

    with open("amd_img_hits.json", 'w') as f:
        json.dump(hits, f)

    with open("amd_img_fails.json", 'w') as f:
        json.dump(fails, f)

    print "Fetched %.0f%% of images" % (len(hits) * 100.0 / len(results))