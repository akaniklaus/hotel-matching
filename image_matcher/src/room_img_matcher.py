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

from settings import IMAGES_PATH, FEATURES_PATH
from utils import load_matches
from feature_extractor import load_global_features, load_local_features, load_vocab
from collections import defaultdict
import json
import numpy as np
from sklearn.metrics import classification_report
from feature_extractor import load_idfs


class LocalMatcher(object):
    def __init__(self, bkgid):
        self.hog_index = {}
        self.load(join(FEATURES_PATH, './booking/%s/hog_lsh.txt'% bkgid), self.hog_index)

    def load(self, pin, obj):
        for line in open(pin):
            path, f_str, code = line.strip().split('\t')
            f = eval(f_str)
            code = eval(code)
            if code not in obj:
                obj[code] = {} 
            if path not in obj[code]:
                obj[code][path] = []
            obj[code][path].append(f)

    def match(self, img_dst, obj, f_func, h_func, d_func):
        F = f_func(img_dst)
        match_dict = defaultdict(float)
        for f in F:
            code = h_func(f)
            if code not in obj:
                continue
            for path in obj[code]:
                match_dict[path] += 1.
        result_list = [(k, 1-v/len(F)) for k, v in match_dict.items()]
        sort_list = sorted(result_list, key=lambda d:d[1])
        return sort_list[:5]

    def match_feats(self, img_feat, obj, d_func):        
        match_dict = defaultdict(float)
        nfs = 0.0
        for code, fs in img_feat.items():
            nfs += len(fs)
            if code not in obj:
                continue
            for path in obj[code]:
                match_dict[path] += len(fs)
        result_list = [(k, 1-v/nfs) for k, v in match_dict.items()]
        sort_list = sorted(result_list, key=lambda d:d[1])
        
        return sort_list      

    def find_similar(self, imgfeats):
        if not 'hog_lsh' in imgfeats:
            return []
        hog_list = self.match_feats(imgfeats['hog_lsh'], self.hog_index, eculidean_dist)

        return hog_list

    def search(self, dst_thum, debug=False):
        hog_list = self.match(dst_thum, self.hog_index, hog3, LSH_hog, eculidean_dist)
        #hog_list2 = self.match2(dst_thum, self.hog_index, hog3, LSH_hog, eculidean_dist)
        sift_list = self.match(dst_thum, self.sift_index, sift2, LSH_sift, eculidean_dist)
        #sift_list2 = self.match2(dst_thum, self.sift_index, sift2, LSH_sift, eculidean_dist)
        local_list = blending([(hog_list, 0.8),
                               (sift_list, 1),
                               ], 5, 2)
        local_list2 = ensembling([(hog_list, 1),
                               (sift_list, 1),
                               ], 5, 2)
        if debug:
            return [
                ('hog lsh', hog_list),
                ('sift lsh', sift_list),
                ('similar images (local similarity)', local_list),
                ('local ensembing', local_list2),
                ]
        else:
            return [
                ('similar images (local similarity)', local_list),
                ]


class GlobalMatcher(object):
    def __init__(self, bkgid, gmparams={}):
        # Load params
        self.bow_weight = gmparams.get('bow_weight') or 0.5
        self.rhisto_weight = (1 - self.bow_weight) * (gmparams.get('rhisto_weight') or 0.5)
        self.yhisto_weight = 1 - -self.bow_weight - self.rhisto_weight
        self.th = gmparams.get('th') or None

        self.use_idf = gmparams.get('use_idf') or False
        if self.use_idf:
            self.idfs = load_idfs()

        self.combine = eval(gmparams.get('combine')) or ensembling

        # Load image features
        self.phash = {}
        self.load(join(FEATURES_PATH, './booking/%s/phash.txt'% bkgid), self.phash)

        self.ohash = {}
        self.load(join(FEATURES_PATH, './booking/%s/otsu_hash.txt'% bkgid), self.ohash)
        
        self.ohash2 = {}
        self.load(join(FEATURES_PATH, './booking/%s/otsu_hash2.txt'% bkgid), self.ohash2)

        self.rgbhisto = {}
        self.load(join(FEATURES_PATH, './booking/%s/rgb_histo.txt'% bkgid), self.rgbhisto)

        self.yuvhisto = {}
        self.load(join(FEATURES_PATH, './booking/%s/yuv_histo.txt'% bkgid), self.yuvhisto)

        self.bow = {}
        self.load(join(FEATURES_PATH, './booking/%s/bow.txt' % bkgid), self.bow)

    def load(self, pin, obj):
        for line in open(pin):
            try:
                feat_name = pin.split('/')[-1].split('.')[0]
                path, hcode = line.strip().split('\t')
                val = eval(hcode)
                if feat_name == 'bow':
                    val = np.array(val)
                    if self.use_idf:
                        val *= self.idfs
                        val /= np.linalg.norm(val)
                obj[path] = val
            except Exception, e:
                print repr(e)

    def match(self, img_path, obj, f_func, d_func):
        code = f_func(img_path)
        value_list = []
        for path in obj:
            d = d_func(obj[path], code)
            value_list.append((path, d))
        sort_list = sorted(value_list, key=lambda d:d[1])
        return sort_list[:5]

    def match_feats(self, img_feat, obj, d_func, is_bow=False):
        """
            Same as match but it uses precomputed features
        """
        code = img_feat
        if is_bow and self.use_idf:
            code *= self.idfs
            code /= np.linalg.norm(code)        
        value_list = []
        for path in obj:
            d = d_func(obj[path], code)
            value_list.append((path, d))
        sort_list = sorted(value_list, key=lambda d:d[1])
        
        return sort_list

    def find_duplicates(self, imgfeats):
        phash_list = self.match_feats(imgfeats['phash'], self.phash, hamming)
        ohash_list = self.match_feats(imgfeats['otsu_hash'], self.ohash, hamming)
        hashes_list = [(phash_list, 1), (ohash_list, 1)]

        if 'otsu_hash2' in imgfeats:
            ohash_list2 = self.match_feats(imgfeats['otsu_hash2'], self.ohash2, hamming)
            hashes_list.append((ohash_list2, 1))
        
        duplicates_list = blending(hashes_list, 1, 6)

        return duplicates_list

    def find_similar(self, imgfeats, nresults=20):
        rhisto_list = self.match_feats(imgfeats['rgb_histo'], self.rgbhisto, chi2_dist)
        yhisto_list = self.match_feats(imgfeats['yuv_histo'], self.yuvhisto, chi2_dist)
        bow_list = self.match_feats(imgfeats['bow'], self.bow, eculidean_dist, is_bow=True)

        combined_list = self.combine([
                               (rhisto_list, self.rhisto_weight),
                               (yhisto_list, self.yhisto_weight),
                               (bow_list, self.bow_weight)
                               ], nresults, 2)

        if self.th:
            combined_list = [(im, d) for (im, d) in combined_list if d < self.th]

        return combined_list


    def search(self, img_path, debug=False):
        phash_list = self.match(img_path, self.phash, phash, hamming)
        ohash_list = self.match(img_path, self.ohash, otsu_hash, hamming)
        ohash_list2 = self.match(img_path, self.ohash2, otsu_hash2, hamming)
        hash_list = blending([(phash_list, 1), 
                              (ohash_list, 1),
                              (ohash_list2, 1),
                              ], 1, 6)

        rhisto_list = self.match(img_path, self.rgbhisto, rgb_histo, abs_dist)
        yhisto_list = self.match(img_path, self.yuvhisto, yuv_histo, abs_dist)
        histo_list = blending([
                               (rhisto_list, 1),
                               (yhisto_list, 1),
                               ], 5, 2)
        histo_list2 = ensembling([
                               (rhisto_list, 1),
                               (yhisto_list, 1),
                               ], 5, 6)

        if debug:
            return [
                    ('duplicate images', hash_list), 
                    ('rgb', rhisto_list),
                    ('yuv', yhisto_list),
                    ('similar images (color histogram)', histo_list), 
                ]
        else:
            return [
                    ('duplicate images', hash_list), 
                    ('similar images (color histogram)', histo_list), 
                ]


def image_matches_for_hotel_match(amdid, bkgid, gmparams=None):
    gm = GlobalMatcher(bkgid, gmparams)

    amdpath = join(FEATURES_PATH, './amadeus/%s/' % amdid)
    if not exists(amdpath):
        return {}, {}

    duplicates = {}
    similar = {}

    gfeats = load_global_features(amdpath)
    for imgpath in gfeats.keys():
        img_gf = gfeats[imgpath]
        if not 'otsu_hash2' in img_gf:
            print "otsu_hash2 missing for %s" % imgpath

        imgdups = gm.find_duplicates(img_gf)               
        if imgdups:
            duplicates[imgpath] = [x[0] for x in imgdups]
        else:
            imgsims = gm.find_similar(img_gf)

            if imgsims:
                similar[imgpath] = imgsims    

    return duplicates, similar

def image_matches_for_single_image(imgpath, bkgid=None, gweight=0.9, gmparams=None):
    amdid = imgpath.split('/')[7]
    if bkgid is None:
        bkgid = load_matches()[amdid]

    gm = GlobalMatcher(bkgid, gmparams)
    lm = LocalMatcher(bkgid)

    amdpath = join(FEATURES_PATH, './amadeus/%s/' % amdid)

    gfeats = load_global_features(amdpath)
    lfeats = load_local_features(amdpath)
    
    img_lf = lfeats[imgpath]
    img_gf = gfeats[imgpath]

    global_list = gm.find_similar(img_gf)
    local_list = lm.find_similar(img_lf)

    combined_list = ensembling([(global_list, gweight), (local_list, 1 - gweight)])

    return combined_list

def evaluate(gmparams):
    with open('img_matches_good.json') as f:
        goodm = json.load(f)
    with open('img_matches_bad.json') as f:
        badm = json.load(f)

    sims = {}
    pairs = goodm.items() + badm.items()
    for imgpath, bkgpath in pairs:
        bkgid = int(bkgpath.split('/')[7])
        try:
            sims[imgpath] = image_matches_for_single_image(imgpath, bkgid, 1.0, gmparams)
        except Exception as e:
            pass

    # Remove those with missing sims
    goodm = {k: v for k,v in goodm.items() if k in sims}
    badm = {k: v for k,v in badm.items() if k in sims}
    pairs = goodm.items() + badm.items()

    y_true = np.array([1] * len(goodm) + [0] * len(badm))

    # compare thresholds
    max_accuracy = 0
    best_th = None
    best_y_pred = None
    if (gmparams.get('combine') != 'blending'):
        th_range = np.arange(0.4, 1.0, 0.025)
    else:
        th_range = np.arange(0.005, 0.05, 0.0025)

    for th in th_range:
        matches = {}
        for imgpath, simlist in sims.items():
            matches[imgpath] = [bpath for bpath, d in simlist if d < th] 

        y_pred = np.zeros(len(pairs))
        for i, (a, b) in enumerate(pairs):
            y_pred[i] = b in matches[a]

        accuracy = sum(y_true == y_pred) * 1.0 / len(pairs)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_th = th
            best_y_pred = y_pred

    # print classification report for good/bad matches
    print "Best accuracy: %.1f%% for th %.2f" % (100 * max_accuracy, best_th)
    print classification_report(y_true, best_y_pred)

def grid_search():
    # grid = [
    #     {'bow_weight': w, 'use_idf': v} for w in [0.36, 0.37, 0.38, 0.39] for v in [True, False]
    # ]

    # grid = [
    #     {'bow_weight': 0.36, 'use_idf': True, 'combine': c, 'rhisto_weight': r} \
    #     for c in ['ensembling', 'blending'] for r in np.arange(0, 0.55, 0.06)
    # ]

    grid = [
        {'bow_weight': b, 'use_idf': True, 'combine': 'ensembling', 'rhisto_weight': r} \
        for r in np.arange(0.5, 0.91, 0.1)\
        for b in np.arange(0.3, 0.61, 0.05)
    ]


    grid = [
        {'bow_weight': b, 'use_idf': True, 'combine': 'blending', 'rhisto_weight': r} \
        for r in np.arange(0.128, 0.132, 0.001)\
        for b in np.arange(0.008, 0.013, 0.001)
    ]

    for gmparams in grid:
        print "Best threshold for %s" % repr(gmparams)
        evaluate(gmparams)

# Best threshold for 

# {'use_idf': True, 'rhisto_weight': 0.144, 'combine': 'blending', 'bow_weight': 0.018}

# Best accuracy: 88.4% for th 0.01
#              precision    recall  f1-score   support

#           0       0.85      0.89      0.87        37
#           1       0.91      0.88      0.90        49

# avg / total       0.89      0.88      0.88        86



if __name__ == '__main__':
    matches = load_matches()
    pairs = matches.items()

    duplicates = {}
    similar = {}
    gmparams = {'use_idf': True, 'rhisto_weight': 0.144, 
                'combine': 'blending', 'bow_weight': 0.018,
                'th': 0.01
    }

    for amdid, bkgid in pairs[:10]:
        dups, sims = image_matches_for_hotel_match(amdid, bkgid, gmparams)
        similar.update(sims)
        duplicates.update(dups)

    with open('duplicates.json', 'w') as f:
        json.dump(duplicates, f)

    with open('similar.json', 'w') as f:
        json.dump(similar, f)
