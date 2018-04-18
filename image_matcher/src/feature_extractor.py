from os.path import exists, join
import os
from shutil import rmtree

from pycbir.util.img_hash import EXTS, phash, otsu_hash, otsu_hash2, hamming 
from pycbir.util.img_histo import rgb_histo, yuv_histo, abs_dist
from pycbir.util.img_hog import hog3, hog_histo, hog_lsh_list
from pycbir.util.lsh import LSH_hog, LSH_sift
from pycbir.util.img_sift import sift2, sift_lsh_list, sift_histo


from settings import IMAGES_PATH
from settings import *

from utils import create_dir_if_needed, timeit, format_time, load_matches
import time
import json
from multiprocessing import Pool
from collections import defaultdict
from scipy.spatial import distance
from scipy.stats import itemfreq
import numpy as np
import pickle

NPROCESSES = 64

def prepare(hotelpath, feat_name):
    print "Computing %s for %s" % (feat_name, hotelpath)
    start = time.time()

    func = eval(feat_name)

    hotel_feats_path = hotelpath.replace('/images/', '/features/')
    create_dir_if_needed(hotel_feats_path)

    p_out = join(hotel_feats_path, '%s.txt' % feat_name)
    with open(p_out, 'w') as f_out:
        for root, dirs, files in os.walk(hotelpath):
            for f in files:
                postfix = f.split('.')[-1]
                if postfix not in EXTS: continue
                full_path = join(root, f)
                try:
                    F = func(full_path)
                    if feat_name == 'bow':
                        F = F.tolist()
                    rep = repr(F)
                    f_out.write('%s\t%s\n' % (full_path, rep))
                except Exception, e:
                    print repr(e)
                    print full_path

    elapsed = time.time() - start
    return elapsed

def prepare_local(hotelpath, f_func, h_func, feat_name):
    print "Computing %s for %s" % (feat_name, hotelpath)
    start = time.time()

    p_out = join(hotelpath, '%s.txt' % feat_name)
    with open(p_out, 'w') as f_out:
        for root, dirs, files in os.walk(hotelpath):
            for f in files:
                postfix = f.split('.')[-1]
                if postfix not in EXTS: continue
                full_path = join(root, f)
                try:
                    F = f_func(full_path)
                    for f in F:
                        f = list(f)
                        h = h_func(f)
                        f_out.write('%s\t%s\t%s\n' % (full_path, repr(f), repr(h)))
                except Exception, e:
                    print repr(e)
                    print full_path

    elapsed = time.time() - start
    return elapsed

def prepare_hotel(hotelpath):
    print "Processing %s" % hotelpath
    times = {}
    if not exists(hotelpath):
        print "Missing images for %s" % hotelpath
        return times

    hotel_feats_path = hotelpath.replace('images', 'features')
    for feat_name in ['phash', 'otsu_hash', 'otsu_hash2',\
                      'rgb_histo', 'yuv_histo', 'bow']:
        feat_time = prepare(hotelpath, feat_name)
        times[feat_name] = feat_time

    # prepare(hotelpath, gist, 'gist')
    # times['hog_lsh'] = prepare_local(hotelpath, hog3, LSH_hog, 'hog_lsh')
    # times['sift_lsh'] = prepare_local(hotelpath, sift2, LSH_sift, 'sift_lsh')

    return times

def load_global_features(hotelpath):
    features = defaultdict(dict)
    for feat_name in ['phash', 'otsu_hash', 'otsu_hash2',\
                      'rgb_histo', 'yuv_histo', 'bow']:
        feats_path = join(hotelpath, '%s.txt' % feat_name)
        if exists(feats_path):
            with open(feats_path) as f:
                for line in f:
                    imgpath, val = line.split('\t')
                    features[imgpath][feat_name] = eval(val)

    return features

def load_local_features(hotelpath):
    features = defaultdict(dict)
    for feat_name in ['hog_lsh']:
        feats_path = join(hotelpath, '%s.txt' % feat_name)
        with open(feats_path) as f:
            for line in f:
                imgpath, f_str, code = line.strip().split('\t')
                f = eval(f_str)
                code = eval(code)
                if not feat_name in features[imgpath]:
                    features[imgpath][feat_name] = defaultdict(list)
                features[imgpath][feat_name][code].append(f)                

    return features


# SIFT-based Visual TF-IDF

VOCAB_FILE = 'vocab.pickle'
def load_vocab():
    with open(VOCAB_FILE, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

VOCAB = load_vocab()

IDFS_FILE = 'idfs.pickle'
def build_idfs(desc_corpus, vocab_km):
    """
        Given a corpus of visual descriptors and
        a fitted KMeans vocabulary,
        it computes the array of inverse document frequencies
        of the corpus relative to the vocabulary
    """
    n_clusters = vocab_km.n_clusters
    dfs = np.zeros(n_clusters, dtype=int)
    for w in vocab_km.predict(desc_corpus):
        dfs[w] += 1 

    idfs = np.log(len(desc_corpus) / (dfs + 2**-23))
    with open(IDFS_FILE, 'wb') as f:
        pickle.dump(idfs, f)

    return idfs

def load_idfs():
    with open(IDFS_FILE, 'rb') as f:
        idfs = pickle.load(f)
    return idfs

def sift_feats_to_norm_bow(desc, vocab):
    """
        Given a matrix of visual descriptors ( e.g.: SIFT ) of an image
        and a vocabulary of centroids of descriptor clusters ( computed with K-means ),
        it computes the term frequencies relative to that vocabulary
    """
    dist2 = distance.cdist(desc, vocab, metric='sqeuclidean')
    assignments = np.argmin(dist2, axis=1)
    idx, count = np.unique(assignments, return_counts=True)

    bow = np.zeros(len(vocab), dtype=float)
    for i, c in zip(idx, count):
        bow[i] = c

    bow /= np.linalg.norm(bow)

    return bow

def bow(imgpath):
    s = sift2(imgpath)
    return sift_feats_to_norm_bow(s, VOCAB)

if __name__ == '__main__':
    matches = load_matches()

    start = time.time()
    hpaths = []

    for amdid, bkgid in matches.items():
        hpaths.append(join(IMAGES_PATH, 'booking/%d' % bkgid))
        hpaths.append(join(IMAGES_PATH, 'amadeus/%s' % amdid))

    p = Pool(NPROCESSES)
    htimes = p.map(prepare_hotel, hpaths)

    elapsed = time.time() - start

    global_times = defaultdict(float)    
    for times in htimes:
        for f, v in times.items():
            global_times[f] += v

    for f in global_times.keys():
        global_times[f] = global_times[f] / len(hpaths)

    print global_times

    nhotels = len(hpaths)
    print "# Hotels: %d" % nhotels
    
    thotel = sum(global_times.values())
    print "Time per hotel: %.2f secs" % thotel
    print "Processing time: %s" % format_time( thotel * nhotels )
