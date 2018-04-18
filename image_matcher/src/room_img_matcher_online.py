from pycbir.util.prepare import prepare_all
import pymongo
from os.path import join, exists
import os
from urllib import urlretrieve
import eventlet
from eventlet.green import urllib

from pycbir.util.img_hash import EXTS, phash, otsu_hash, otsu_hash2, hamming 
from pycbir.util.img_histo import gray_histo, rgb_histo, yuv_histo, hsv_histo, abs_dist, chi2_dist
from pycbir.util.img_gist import gist

from pycbir.util.kmeans import eculidean_dist, norm0_dist
from pycbir.util.rerank import blending, ensembling

from settings import IMAGES_PATH
from utils import load_matches
from feature_extractor import load_global_features, load_local_features, load_vocab
from collections import defaultdict
import json
import numpy as np
from sklearn.metrics import classification_report

from feature_extractor import *
from room_img_matcher import *
from img_downloader_bkg import fetch_room_images, get_conn_bkg, save_img as save_img_bkg
from img_downloader_amd import fetch_img_download_data, get_conn_amd, save_img as save_img_amd

from multiprocessing import Pool, Manager, Process, log_to_stderr
import logging
mpl = log_to_stderr()
mpl.setLevel(logging.INFO)


NPROCESSES = 64

def fetch_img_features_bkg(bkgid, c):
    # download images
    c = c['booking']['details']    
    ufs = fetch_room_images(c, bkgid, update=True)

    path_to_url = {p: u for (u,p) in ufs}

    pool = eventlet.GreenPool()
    for u, succeeded in pool.imap(save_img_bkg, ufs):
        if succeeded:
            print "got img from %s" % u
        else:
            print "failed to get img for %s" % u

    # get features
    hpath = join(IMAGES_PATH, 'booking/%d' % bkgid)
    prepare_hotel(hpath)

    # delete images
    if exists(hpath):
        rmtree(hpath)

    # return path to url mapping
    return path_to_url


def fetch_img_features_amd(amdid, conn, counts):
    path_to_url = {}

    # download images
    results = fetch_img_download_data(conn, [amdid])
    counts['listed'] += len(results)

    pool = eventlet.GreenPool()
    for i, u, fpath, succeeded in pool.imap(save_img_amd, results):
        if succeeded:
            path_to_url[fpath] = u
            counts['downloaded'] += 1
            print "%s: got img from %s" % (i, u)
        else:
            print "%s: failed to get img from %s" % (i, u)

    # get features
    hpath = join(IMAGES_PATH, 'amadeus/%s' % amdid)
    prepare_hotel(hpath)

    # delete images
    if exists(hpath):
        rmtree(hpath)

    # return path to url mapping
    return path_to_url


def image_matches_for_hotel_match(amdid, bkgid, conn_amd, conn_bkg, gmparams, counts):
    # Fetch image features from online images
    path_to_url = fetch_img_features_bkg(bkgid, conn_bkg)
    pu = fetch_img_features_amd(amdid, conn_amd, counts)
    path_to_url.update(pu)

    amdpath = join(FEATURES_PATH, './amadeus/%s/' % amdid)

    dups = {}
    sims = {}

    bkgpath = join(FEATURES_PATH, './booking/%s/' % bkgid)
    if not exists(bkgpath):
        return dups, sims

    gm = GlobalMatcher(bkgid, gmparams)

    gfeats = load_global_features(amdpath)

    for imgpath in gfeats.keys():
        img_gf = gfeats[imgpath]
        if not 'otsu_hash2' in img_gf:
            print "otsu_hash2 missing for %s" % imgpath

        imgdups = gm.find_duplicates(img_gf)               
        if imgdups:
            counts['matched'] += 1            
            for x in imgdups:
                ipath = x[0]
                room_name = ipath.split('/')[-2]
                dups[path_to_url[imgpath]] = (path_to_url[ipath], room_name)
        else:
            imgsims = gm.find_similar(img_gf)
            if imgsims:
                counts['nearly_matched'] += 1
                url = path_to_url[imgpath]
                sims[url] = []
                for (ipath, d) in imgsims:
                    room_name = ipath.split('/')[-2]
                    sims[url].append(((path_to_url[ipath], room_name), d))

    return dups, sims

gmparams = {'use_idf': True, 'rhisto_weight': 0.144, 
            'combine': 'blending', 'bow_weight': 0.018,
            'th': 0.01
}

def pair_matches(args):
    amdid, bkgid, duplicates, similar, counts = args

    ca = get_conn_amd()
    cb = get_conn_bkg()

    dups, sims = image_matches_for_hotel_match(amdid, bkgid, ca, cb, gmparams, counts)
    pair = "%s %d" % (amdid, bkgid)
    if dups:
        duplicates[pair] = dups
    
    if sims:
        similar[pair] = sims

    ca.close()
    cb.close()



if __name__ == '__main__':
    matches = load_matches()
    pairs = matches.items()

    manager = Manager()

    duplicates = manager.dict()
    similar = manager.dict()
    # Here we will keep counts of Amadeus images:
    # - listed on DB
    # - successfully downloaded
    # - matched ( duplicate found )
    # - nearly mateched ( very similar image found )
    counts = manager.dict()
    counts['listed'] = 0
    counts['downloaded'] = 0
    counts['matched'] = 0
    counts['nearly_matched'] = 0  

    p = Pool(NPROCESSES)

    args = [(amdid, bkgid, duplicates, similar, counts) for (amdid, bkgid) in pairs]
    # pair_matches(args[0])
    p.map(pair_matches, args)

    # p.close()

    with open('duplicates.json', 'w') as f:
        json.dump(dict(duplicates), f)

    with open('similar.json', 'w') as f:
        json.dump(dict(similar), f)

    print dict(counts)