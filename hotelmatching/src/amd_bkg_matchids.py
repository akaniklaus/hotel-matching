#!/usr/bin/python
#-*-coding:utf-8-*-
"""
    Identification of Amadeus hotels
    with Booking.com hotels through
    name similarity and geographical distance
"""
import math
import os
import re
import unicodedata as ud
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
import sys
import codecs
import pymongo
from nltk import FreqDist, tokenize
import pickle
from geoindex import GeoGridIndex, GeoPoint
from itertools import combinations
import psycopg2
import pymysql
import json
import codecs
from collections import defaultdict

MONGODB_SERVER = "52.233.173.236"
MONGODB_PORT = 27017
MONGODB_USER = 'livingrooms:TXrW1IhGzp'

def get_mongo_conn(db_name):
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

connection = get_mongo_conn('booking')

# ENCODING = sys.stdin.encoding
ENCODING = "utf-8"

GLOBAL_STOPWORDS = [
    '&',
    "'s",
    '1',
    '2',
    'a',
    'and',
    'aparthotel',
    'apartment',
    'apartments',
    'at',
    'b',
    'beach',
    'bed',
    'bedroom',
    'best',
    'business',
    'by',
    'city',
    'cottage',
    'de',
    'del',
    'express',
    'guest',
    'guesthouse',
    'historic',
    'home',
    'hostel',
    'hotel',
    'hotels',
    'house',
    'i',
    'in',
    'la',
    'le',
    'les',
    'lodge',
    'motel',
    'on',
    'park',
    'residence',
    'resort',
    'spa',
    'studio',
    'suites',
    'the',
    'view',
    'villa',
    'with'
]

ABBREVIATIONS = {
    'suites': 'ste',
    'stes': 'ste',
    'airport': 'ap',
    'arpt': 'ap',
    'downtown': 'dtwn',
    'dwtwn': 'dtwn',
    'dwtn': 'dtwn',
    'international': 'intl',
    'homewood': 'hmwd',
    'hilton': 'hltn',
    'howard johnson': 'hojo',
    'ii': '2',
    u'ö': 'oe',
    u'ä': 'ae',
    u'ü': 'ue'
}

def json_dump_unicode(data, file_path):
    with codecs.open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_commented_json(fpath):
    with open(fpath) as f:
        lines = [l for l in f.readlines() if l.strip()[:2] != '//']
        j = '\n'.join(lines)
        j = re.sub(r",\s*\}", '}', j)
        return json.loads(j)

def get_amd_db_conn():
    conn = psycopg2.connect(database='d36tike0jg2ieb',
                        user='u4esungn748mp7',
                        password='pb31e2aedb696ceb6dffc2c4f0756fd59ec9838854b739fe027d7cc44a2c13fcb',
                        host='ec2-52-212-211-39.eu-west-1.compute.amazonaws.com',
                        port=5432)
    return conn 

def load_amd_chain_codes():
    conn = get_amd_db_conn()
    query = """select "vendorId" from public."VendorIds" where vendor='amadeus'"""
    cur=conn.cursor()
    cur.execute(query)
    vids = [r for r, in cur.fetchall()]
    hcodes = list(set([v[:2] for v in vids]))

    with open('chain_code_name.json') as f:
        code_chains = json.load(f)
        del code_chains['HS']
        # Note:  properties with the HS chain code (HRS chain) come from HRS database (HRS.de).
        # Although the chain code is HS, they are independent properties
        # that do not belong to a proper chain (like hilton).

    code_chains = {k: v for (k,v) in code_chains.items() if k in hcodes}

    return code_chains

AMD_CODE_TO_CHAIN = load_amd_chain_codes()

def load_amadeus_from_db(amd_codes=None, limit=None):
    """
        Given a dataframe with spreadsheet data from Amadeus hotels,
        loads matching features for Amadeus hotels
        that haven't already been matched to a Booking.com hotel
    """
    conn = get_amd_db_conn()
    cur = conn.cursor()
    query = """SELECT pr.id, pr.position, pr.name, v."vendorId"
               FROM public."Properties" pr
               JOIN public."VendorIds" v
               ON pr.id = v.relation
               WHERE v.vendor = 'amadeus'
            """

    if amd_codes is not None:
        query += """AND v."vendorId" IN (%s)""" %\
                    ', '.join(["'%s'" % x for x in amd_codes])
    if limit:
        query += " LIMIT %d" % limit
    cur.execute(query)

    rows = []
    for r in cur.fetchall():
        lvr_id, position, name, vendorid = r
        name = name.decode('utf8')
        chain = AMD_CODE_TO_CHAIN.get(vendorid[:2]) or ''
        bkg_id = None
        lng, lat = [float(x) for x in position[1:-1].split(',')]
        rows.append((lvr_id, lat, lng, name, chain, bkg_id, vendorid))

    amdh = pd.DataFrame(rows)    
    amdh.columns = ['lvr_id', 'lat', 'lng', 'name', 'chain', 'bkg_id', 'amd_code']
    amdh.index = amdh.lvr_id
    
    return amdh

def load_amadeus_spreadsheet():
    """
        DEPRECATED: we now load from postgresql

        Given a dataframe with spreadsheet data from Amadeus hotels,
        loads matching features for Amadeus hotels
        that haven't already been matched to a Booking.com hotel
    """
    amdh = pd.read_excel('./AmadeusProps-matched.xlsx', header=0, index_col=None)

    amdh = amdh[['DUPE_POOL_ID', 'LATITUDE', 'LONGITUDE', 'PROPERTY_NAME', 'CHAIN_NAME', 'BOOKING_ID']]
    amdh.columns = ['lvr_id', 'lat', 'lng', 'name', 'chain', 'bkg_id']

    # Clean up bad latitude data
    def ensure_length(n):
        if type(n) is int:
            s = str(n)
            if len(s) ==  6:
                s = s + '0'
                return int(s) / (10.0 ** 5)
            elif len(s) == 7:
                return int(s) / (10.0 ** 5)
            elif len(s) == 8:
                return int(s) / (10.0 ** 6)
            else:
                return None
        else:
            return n

    amdh.lat = amdh.lat.apply(ensure_length)
    amdh.lat = amdh.lat.apply(lambda x: np.float64(x) if type(x) is str else x)


    good_inds = [i for i, v in enumerate(amdh.lat.values) if np.isscalar(v)]

    amdh = amdh.iloc[good_inds,:]
    amdh.lng /= (10.0 ** 5)
    
    return amdh

def load_booking(hotel_ids=None):
    """
        Given a dataframe with spreadsheet data from Amadeus hotels,
        loads matching features for Booking.com hotels
        that haven't already been matched to an Amadeus hotel
    """
    # TODO: fetch bkgh data
    det = connection['booking']['details']

    query = { 'hotel_id': { '$in': [str(x) for x in hotel_ids] } } if hotel_ids else {}
    hdata = det.find(query, {'hotel_id': 1, 'lat': 1, 'lng': 1, 'title': 1, 'chain': 1})
    bkgh = []
    for r in hdata:
        hid = int(r['hotel_id'])
        chain = r['chain'][0] if r['chain'] else None
        bkgh.append({'lvr_id': None, 'lat': r['lat'], 'lng': r['lng'], 'name': r['title'], 'bkg_id': hid, 'chain': chain})
    
    bkgh = pd.DataFrame(bkgh)
    bkgh.index = bkgh.bkg_id

    return bkgh 

def load_booking_from_mysql(hotel_ids=None):
    """
        Given a dataframe with spreadsheet data from Amadeus hotels,
        loads matching features for Booking.com hotels
        that haven't already been matched to an Amadeus hotel
    """
    mysql_args = {
        'host': 'localhost',
        'user': 'tuyio',
        'passwd': 'tuyio',
        'db': 'tuyiorates',
        'charset': 'utf8',
        'use_unicode': 1
        # 'query': {'charset': 'utf8', 'use_unicode': 1}
    }

    conn = pymysql.connect(**mysql_args)
    cur = conn.cursor()

    query = """SELECT HotelID, Latitude, Longitude, HotelName, ChainName
           FROM tbl_Hotel
    """

    if hotel_ids:
        query += " WHERE HotelID IN (%s)" % ', '.join([str(x) for x in hotel_ids])

    cur.execute(query)

    bkgh = []
    for row in cur:
        hid, lat, lng, name, chain = row
        hdata = {'lvr_id': None,
                 'lat': float(lat),
                 'lng': float(lng),
                 'name': name,
                 'bkg_id': hid,
                 'chain': chain
        }
        bkgh.append(hdata)
    
    bkgh = pd.DataFrame(bkgh)
    bkgh.index = bkgh.bkg_id

    cur.close()
    conn.close()

    return bkgh 

def load_matches():
    try:
        # with open('matches.pickle','rb') as f:
        #     matches = pickle.load(f)
        with open('matches.json') as f:
            matches = json.load(f)
    except IOError:
        matches = {}
    return matches

def get_chain_matches(amdh, bkgh, matches):
    chain_matches = {}
    _matches = {a:b for a,b in matches.items() if b in bkgh.bkg_id.values}
    for _, a in amdh.iterrows():
        lvr_id = a['lvr_id']
        if lvr_id not in _matches:
            continue
        bkg_id = _matches[lvr_id]

        chain_a = a['chain']
        if not chain_a:
            chain_a = "NONE"

        chain_b = bkgh.loc[bkg_id, 'chain']
        if not chain_b:
            chain_b = "NONE"
        else:
            chain_b = chain_b[0]
        
        if chain_a not in chain_matches:
            chain_matches[chain_a] = defaultdict(int)
        chain_matches[chain_a][chain_b] += 1

    json_dump_unicode(chain_matches, 'chain_matches.json')
    # with open('chain_matches.json','w') as f:
    #     json.dump(chain_matches,f)

    return chain_matches

def dist_to_sim(d):
    """
        Maps a distance in [0, ∞) interval
        to a similarity score in (0, 1]

        We use an exponential with base b such that
        
        d = 0 mts -> sim = 1
        d = 50 mts -> sim = 0.5
    """
    b = 0.5 ** (1.0/50)
    return b ** d

def get_geo_dist(p1, p2):
    """
        Geographic distance using Haversine formula
    """
    lat1, lng1 = p1
    lat2, lng2 = p2
    r = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    lambda1 = math.radians(lng1)
    lambda2 = math.radians(lng2)
    haversin = lambda theta: (1 - math.cos(theta)) / 2.0
    h = haversin(phi2 - phi1) + math.cos(phi1) * math.cos(phi2) * \
        haversin(lambda2 - lambda1)
    c = 2 * math.atan2(math.sqrt(h), math.sqrt(1 - h))
    return r * c

def remove_accents(s):
    return ''.join((c for c in ud.normalize('NFD', s) if ud.category(c) != 'Mn'))

def extract_stopwords(hotel_names):
    nhotels = len(hotel_names)
    hotel_names = [normalize(h) for h in hotel_names if type(h) is str]
    words = tokenize.word_tokenize(' '.join(hotel_names))
    fdist = FreqDist(words)
    stopwords = [x[0] for x in fdist.most_common(20) if x[1] > nhotels / 3]

    return stopwords

def normalize(hname, local_stopwords=[], disable_stopwords=False):
    """
        Split in words, turn to lowercase,
        remove accents
    """
    hname = hname.strip()
    hname = remove_accents(hname).lower()

    separators = "-&/'"
    for s in separators:
        hname = hname.replace(s, ' ')

    hname = ''.join(c for c in hname if c not in ",.:;!|_()[]<>{}\"")

    for w, ab in ABBREVIATIONS.items():
        hname = hname.replace(w, ab)

    tokens = hname.split()
    if not disable_stopwords:
        tokens = [t for t in tokens if t not in GLOBAL_STOPWORDS + local_stopwords]
    
    return ' '.join(tokens)

NAME_SIMS = {}

NAME_SIMS_SW = {}

def get_name_sim(hname1, hname2, swap_words=False, sw=[]):
    """
        Normalize both names
        and obtain string similarity
    """
    if not swap_words:
        hname1 = normalize(hname1, sw)
        hname2 = normalize(hname2, sw)
        if not (hname1, hname2) in NAME_SIMS:
            NAME_SIMS[(hname1, hname2)] = SequenceMatcher(a=hname1, b=hname2).ratio()
        return NAME_SIMS[(hname1, hname2)]
    else:
        hname1 = normalize(hname1, disable_stopwords=True)
        hname2 = normalize(hname2, disable_stopwords=True)

        if not (hname1, hname2) in NAME_SIMS_SW:
            tokens1 = hname1.split()
            tokens2 = hname2.split()

            if len(tokens1) == len(tokens2):
                hname1 = ' '.join(sorted(tokens1))
                hname2 = ' '.join(sorted(tokens2))
                sim = SequenceMatcher(a=hname1, b=hname2).ratio()
            else:
                longest, shortest = (tokens1, tokens2) if len(tokens1) > len(tokens2) else (tokens2, tokens1)
                hname_short = ' '.join(sorted(shortest))
                sim = 0
                # some words might be split differently
                combs = list(combinations(longest, len(shortest))) + list(combinations(longest, len(shortest) + 1))
                if len(shortest) > 1:
                    combs += list(combinations(longest, len(shortest) - 1))
                for tokens in combs:
                    sim = max(sim, SequenceMatcher(a=hname_short, b=' '.join(sorted(tokens))).ratio())

            NAME_SIMS_SW[(hname1, hname2)] = sim
        
        return NAME_SIMS_SW[(hname1, hname2)]

def get_geoname_sim(hotel1, hotel2, sw=[]):
    """
        Combined location and name similarity
    """
    name_similarity = get_name_sim(hotel1['name'], hotel2['name'], sw)
    h1loc = (hotel1['lat'], hotel1['lng'])
    h2loc = (hotel2['lat'], hotel2['lng'])

    location_similarity = dist_to_sim(get_geo_dist(h1loc, h2loc))
    
    return location_similarity * (name_similarity) ** 2

def match_in_neighborhood(amdh, geo_index, radius, nsim_threshold, matches, namdh,
                            save=True, unique=False, swap_words=False, return_cands=True):
    count = 0
    amdh = amdh[~amdh.lvr_id.isin(matches.keys())]

    if return_cands:
        candidates = {}
    for _, h in amdh.iterrows():
        count += 1
        if count % 1000 == 0:
            progress = count * 100.0 / len(amdh)
            print("%.2f %%" % progress)

        center_point = GeoPoint(h['lat'], h['lng'], ref=h)
        try:
            cands = list(geo_index.get_nearest_points(center_point, radius, 'km'))
        except Exception:
            continue

        cands = [hb.ref for (hb, d) in cands]
        cands = []
        for (hb, d) in cands:
            hb_ = hb.ref
            hb_['dist'] = d
            cands.append(hb)

        cands = [hb for hb in cands if hb['bkg_id'] not in matches.values()]
        if not cands:
            continue

        sw = extract_stopwords([hb['name'] for hb in cands])

        nsims_plain = [get_name_sim(hb['name'], h['name'], False, sw) for hb in cands]
        if swap_words:
            nsims_swap = [get_name_sim(hb['name'], h['name'], True, sw) for hb in cands]
            nsims = nsims_swap
        else:
            nsims = nsims_plain

        inds = [i for i in reversed(np.argsort(nsims)) if nsims[i] > nsim_threshold]

        if inds and (not unique or len(inds) == 1) and (not return_cands):
            best_ind = inds[0]        
            hb = cands[best_ind]
            nsim = nsims[best_ind]
            matches[h['lvr_id']] = hb['bkg_id']

        if return_cands:
            candsh = []
            for i in inds:
                ns = nsims[i]
                hb = cands[i]
                cand = {
                    # 'candidate': cands[i],
                    # 'name_sim': ns
                    'lvr_id': h['lvr_id'],
                    'bkg_id': hb['bkg_id'],
                    'name': h['name'],
                    'chain': h['chain'],
                    'name_bkg': hb['name'],
                    'chain_bkg': hb['chain'],
                    'name_sim': nsims_plain[i],
                    'name_sim_sw': nsims_swap[i],
                    'dist': hb['d']
                }
                candsh.append(cand)
            candidates[h['lvr_id']] = candsh
        
    perc_matched = len(matches) * 100.0 / namdh
    print("%.1f%% matched" % perc_matched)

    # Save Matches
    if save:
        with open('matches.json', 'w') as f:
            json.dump(matches, f)

    if return_cands:
        return candidates

def extract_candidates(amdh, geo_index, radius, nsim_threshold, namdh):
    count = 0

    candidates = {}
    for _, h in amdh.iterrows():
        count += 1
        if count % 1000 == 0:
            progress = count * 100.0 / len(amdh)
            print("%.2f %%" % progress)

        center_point = GeoPoint(h['lat'], h['lng'], ref=h)
        try:
            geo_cands = list(geo_index.get_nearest_points(center_point, radius, 'km'))
        except Exception:
            continue

        cands = []
        for (hb, d) in geo_cands:
            hbd = hb.ref
            hbd['dist'] = d
            cands.append(hbd)

        if not cands:
            continue

        sw = extract_stopwords([hb['name'] for hb in cands])

        nsims_plain = [get_name_sim(hb['name'], h['name'], False, sw) for hb in cands]
        nsims_swap = [get_name_sim(hb['name'], h['name'], True, sw) for hb in cands]

        nsims = nsims_swap
        inds = [i for i in reversed(np.argsort(nsims)) if nsims[i] > nsim_threshold]

        candsh = []
        for i in inds:
            ns = nsims[i]
            hb = cands[i]
            cand = {
                # 'candidate': cands[i],
                # 'name_sim': ns
                'lvr_id': h['lvr_id'],
                'bkg_id': hb['bkg_id'],
                'name': h['name'],
                'chain': h['chain'],
                'name_bkg': hb['name'],
                'chain_bkg': hb['chain'],
                'name_sim': nsims_plain[i],
                'name_sim_sw': nsims_swap[i],
                'dist': hb['dist']
            }
            candsh.append(cand)
        candidates[h['lvr_id']] = candsh
        
    return candidates

def generate_comparison_vectors(amdh, geo_index, radius=6, nsim_threshold=0.5):
    """
        Creation of comparison vectors for training a matching classifier

        amdh: data frame containing Amadeus Hotels
        geo_index: index of Booking hotels, easy to query by geo radius
        radius: distance to fetch candidates from
        nsim_threshold: minimum name similarity to be considered a candidate
    """
    cand_pair_ids = []
    features = []

    count = 0
    for i, h in amdh.iterrows():
        count += 1
        if count % 500 == 0:
            progress = count * 100.0 / len(amdh)
            print("%.2f %%" % progress)

        center_point = GeoPoint(h['lat'], h['lng'], ref=h)
        try:
            cands = list(geo_index.get_nearest_points(center_point, radius, 'km'))
        except Exception:
            continue

        nsims1 = [get_name_sim(hb.ref['name'], h['name'], swap_words=False) for (hb, d) in cands]
        nsims2 = [get_name_sim(hb.ref['name'], h['name'], swap_words=True) for (hb, d) in cands]
        inds = [ind for ind in range(len(nsims2)) if nsims2[ind] > nsim_threshold]

        for ind in inds:
            hb, d = cands[ind]
            cand_pair_ids.append((h['lvr_id'], hb.ref['bkg_id']))
            features.append((nsims1[ind], nsims2[ind], d))

    cand_pairs = pd.MultiIndex.from_tuples(cand_pair_ids, names=['lvr_id', 'bkg_id'])
    cand_data = pd.DataFrame(features, index=cand_pairs, columns=['nsim1','nsim2','dist'])

    return cand_data

def generate_match_pairs():
    """
        Generation of multiindex of previously matched pairs,
        to be used for training classifiers
    """
    matches = load_matches()
    match_pairs = pd.MultiIndex.from_tuples(matches.items(), names=['lvr_id', 'bkg_id'])

    return match_pairs

def initialize_matching(overwrite=False):
    # 1. fetch all unmatched Amadeus hotels
    print("Loading Amadeus hotels")
    amdh = load_amadeus_from_db()
    namdh = len(amdh)
    print("Loaded %d hotels" % namdh)

    # 2. fetch all unmatched Booking.com hotels
    print("Loading Booking hotels")
    bkgh = load_booking()
    # bkgh = load_booking_from_mysql()
    print("Loaded %d hotels" % len(bkgh))

    # 3. load existing matches
    if not overwrite:
        print("Loading previous matches")
        matches = load_matches()
    else:
        matches = {}

    matched_amdids = matches.keys()
    matched_bkgids = set(matches.values())

    # 4. Exclude already matched
    amdh = amdh[~amdh.lvr_id.isin(matched_amdids)]
    bkgh = bkgh[~bkgh.bkg_id.isin(matched_bkgids)]
    print "%d Amadeus hotels left to match to %d Booking hotels" % (len(amdh), len(bkgh))

    # 5. Build geo index
    print("Building Geo Index")
    geo_index = GeoGridIndex()
    for i, hb in bkgh.iterrows():
        if hb['lat'] == 90:
            hb['lat'] = -90.0
        geo_index.add_point(GeoPoint(hb['lat'], hb['lng'], ref=hb))

    return amdh, bkgh, matches, geo_index, namdh

def automatic_matching(overwrite=False):
    amdh, bkgh, matches, geo_index, namdh = initialize_matching(overwrite)

    # 6. four passes of radius/name matching
    print("1st pass")
    match_in_neighborhood(amdh, geo_index, 1, 0.6, matches, namdh)

    print("2nd pass")
    match_in_neighborhood(amdh, geo_index, 2, 0.75, matches, namdh)
    
    print("3rd pass")
    match_in_neighborhood(amdh, geo_index, 4, 0.8, matches, namdh)
    
    print("4th pass")
    match_in_neighborhood(amdh, geo_index, 6, 0.86, matches, namdh)

    print("5th pass")
    match_in_neighborhood(amdh, geo_index, 0.1, 0.5, matches, namdh, unique=True)

    # 7. search for matches where words are in different order
    match_in_neighborhood(amdh, geo_index, 6, 0.86, matches, namdh, unique=True, swap_words=True, return_cands=True)

    # 8. Analyze chain names in current matches

    # Interactive matching

def automatic_matching_with_classifier():
    amdh, bkgh, matches, geo_index, namdh = initialize_matching(overwrite)

    candidates = match_in_neighborhood(amdh, geo_index, 6, 0.6, matches, namdh,
                                        save=False, unique=False, swap_words=True, return_cands=True)

def get_interactive_candidates(amdh, geo_index, radius=6, nsim_threshold=0.5):
    """
        Gets candidates for interactive matching for yet unmatched Amadeus hotels
    """
    cands_by_hotel = defaultdict(list)

    for i, h in amdh.iterrows():
        center_point = GeoPoint(h['lat'], h['lng'], ref=h)
        try:
            cands = list(geo_index.get_nearest_points(center_point, radius, 'km'))
        except Exception:
            continue

        nsims = [get_name_sim(hb.ref['name'], h['name'], swap_words=True) for (hb, d) in cands]
        inds = [ind for ind in range(len(nsims)) if nsims[ind] > nsim_threshold]
        inds = sorted(inds, key=lambda i:-nsims[i])

        for ind in inds:
            hb, d = cands[ind]
            cands_by_hotel[h['lvr_id']].append((hb.ref['bkg_id'], nsims[ind], d))

    return cands_by_hotel

def interactive_matching():
    amdh, bkgh, matches, geo_index = initialize_matching()

    # Load candidates
    print "Getting unmatched candidates"
    cands_by_hotel = get_interactive_candidates(amdh, geo_index)

    for amdid, cands in cands_by_hotel.items():
        # remove already matched
        cands = [x for x in cands if x[0] not in matches.values()]

        top_bkgids = [m[0] for m in cands[:5]]
        top_matches = bkgh.loc[top_bkgids,:][['name','chain','lat','lng']]
        list_numbers = [str(x) for x in range(1, len(top_matches) + 1)]
        top_matches['list number'] = list_numbers 

        amd = amdh.loc[amdid,:][['name','chain','lat','lng']]

        os.system('cls' if os.name == 'nt' else 'clear')
        print amd

        print "\nWhich of the following is the best match for the previous Amadeus hotel?"
        print top_matches

        answer = None
        while not answer in list_numbers + ['s']:
            answer = raw_input("\nChoose the list number of the best match or 's' to skip\n")

        if answer != 's':
            answer_i = int(answer) - 1
            bkgid = top_bkgids[answer_i]
            matches[amdid] = bkgid

    with open('matches_interactive.pickle', 'wb') as f:
        pickle.dump(matches, f)

if __name__ == '__main__':
    # automatic_matching()
    interactive_matching()