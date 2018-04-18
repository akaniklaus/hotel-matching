import json
from collections import defaultdict, Counter, OrderedDict
from itertools import chain
import psycopg2
import numpy as np
import pymongo
from progressbar import ProgressBar, Percentage

def get_conn_amd():
    c = psycopg2.connect(database='d36tike0jg2ieb',
                        user='u4esungn748mp7',
                        password='pb31e2aedb696ceb6dffc2c4f0756fd59ec9838854b739fe027d7cc44a2c13fcb',
                        host='ec2-52-212-211-39.eu-west-1.compute.amazonaws.com',
                        port=5432)
    return c

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
    c = pymongo.MongoClient(uri)

    return c['booking']['details']

def get_most_common(l):
    c = Counter(l)
    return [w for w, n in c.most_common() if n == c.most_common(1)[0][1]]

def concatenate(lists):
    return list(chain(*lists))

def tokenize(s):
    return [x.strip() for x in s.split()]

def rt_description_to_amd_rtname(rt_desc):
    rts = set()
    rt_desc = rt_desc.lower()
    kw_map = {
        'studio': 'Studio',
        'apartment': 'Residential apartment',
        'business': 'Business room',
        'standard': 'Standard',
        'superior': 'Superior',
        'penthouse': 'Penthouse',
        'deluxe': 'Deluxe Room',
        'executive': 'Executive room',
        'king room': 'Executive room',
        'queen room': 'Executive room',
        'king suite': 'Concierge/Executive Suite',
        'queen suite': 'Concierge/Executive Suite',
        'conference suite': 'Concierge/Executive Suite',
        'comfort': 'Comfort room',
        'family': 'Family Room',
        'accessible': 'Accessible room',
        'presidential': 'Penthouse',
        'basic': 'Budget room',
        'classic': 'Budget room'

    }
    for kws, rt in kw_map.items():
        if all(kw in tokenize(rt_desc) for kw in kws.split()):
            rts.add(rt)
    if 'suite' in rt_desc and 'Concierge/Executive Suite' not in rts:
        rts.add('Junior Suite/Mini Suite')
    if not rts:
        rts.add('Standard')

    return list(rts)

AMD_TYPECODE_TO_ROOMNAME = OrderedDict([
    ('H', 'Accessible room'),
    ('I', 'Budget room'),
    ('B', 'Business room'),
    ('G', 'Comfort room'),
    ('D', 'Deluxe Room'),
    ('X', 'Duplex'),
    ('E', 'Executive room'),
    ('C', 'Concierge/Executive Suite'),
    ('F', 'Family Room'),
    ('S', 'Junior Suite/Mini Suite'),
    ('P', 'Penthouse'),
    ('R', 'Residential apartment'),
    ('M', 'Standard'),
    ('L', 'Studio'),
    ('A', 'Superior'),
    ('V', 'Villa')
])

def roomtype_ranking(rt_name):
    return AMD_TYPECODE_TO_ROOMNAME.values().index(rt_name)

AMD_ROOMNAMES = AMD_TYPECODE_TO_ROOMNAME.values()

def extract_amd_rt_names(typecode, name, description):
    if name in AMD_ROOMNAMES:
        return [name]
    
    elif typecode:
        name = AMD_TYPECODE_TO_ROOMNAME.get(typecode[0])
        if name:
            return [name]
    else:
        desc = description.split('\n')[0].strip()
        if desc:
            desc_names = rt_description_to_amd_rtname(desc)
            if desc_names:
                return desc_names
        else:
            return []


def inspect_roomtypes_for_matching_pair(amd_id, bkg_id):
    # amd_id=no_matching_bkg_roomtypes[9];bkg_id=matches[amd_id]
    h = conn_bkg.find_one({ 'hotel_id': str(bkg_id) })
    bkg_names = [rt_description_to_amd_rtname(r['name']) for r in h['rooms']]

    query = """
                SELECT rt.id, "typeCode", name, td.text
                FROM public."RoomTypes" rt
                JOIN public."TextDescriptions" td ON td.relation = rt.id
                WHERE property = '%s'
            """ % amd_id
    cur_amd.execute(query)

    rt_livingscores = defaultdict(list)

    amd_rtid_names = {}
    for rt_id, typecode, name, description in cur_amd.fetchall():
        rt_names = extract_amd_rt_names(typecode, name, description)
        if rt_names:
            amd_rtid_names[rt_id] = rt_names

    print bkg_names
    print amd_rtid_names


def assign_amd_roomtype_livingscores():
    # Use hotel matches to generate structure
    # amd_id -> ( roomtype_id -> livingscore )
    # load matches
    with open('matches.json') as f:
        matches = json.load(f)

    amd_rt_livingscores = {}

    conn_amd = get_conn_amd()
    cur_amd = conn_amd.cursor()

    conn_bkg = get_conn_bkg()

    no_amd_rtnames = []
    no_livingscores = []

    # iterate over hotel matches
    bar = ProgressBar(widgets=[Percentage(format='%(percentage)3.2f%%')], max_value=len(matches))
    bar.start()
    for i, (amd_id, bkg_id) in enumerate(matches.items()):
        # Fetch Amadeus room type names
        query = """
                    SELECT rt.id, "typeCode", name, td.text
                    FROM public."RoomTypes" rt
                    JOIN public."TextDescriptions" td ON td.relation = rt.id
                    WHERE property = '%s'
                """ % amd_id
        cur_amd.execute(query)

        rt_livingscores = defaultdict(list)

        amd_rtid_names = {}
        for rt_id, typecode, name, description in cur_amd.fetchall():
            rt_names = extract_amd_rt_names(typecode, name, description)
            if rt_names:
                amd_rtid_names[rt_id] = rt_names
        
        if not amd_rtid_names:
            no_amd_rtnames.append(amd_id)
            continue

        # fetch living scores for rooms 
        # within matched B.com property
        # ( ignore their roomtypes )
        h = conn_bkg.find_one({ 'hotel_id': str(bkg_id) })
        lvs_rooms = [r for r in h['rooms'] if 'livingscore' in r]
        bkg_livingscores = sorted([r['livingscore'] for r in lvs_rooms])

        if not bkg_livingscores:
            no_livingscores.append(amd_id)
            continue

        # Apply livingscores to Amadeus room types
        all_rnames = list(set(concatenate(amd_rtid_names.values())))
        sorted_rn_ranks = sorted([roomtype_ranking(rn) for rn in all_rnames])

        if len(set(all_rnames)) == 1:
            # If there is only one room-type available, we just use
            # the mean livingscore
            lvs = np.mean(bkg_livingscores)
            rt_livingscores = {ri: lvs for ri in amd_rtid_names.keys()} 
        else:
            # if there's more than one room-type then use piecewise linear
            # interpolation as follows:

            # We have K rooms that we need to find its closest living-score. 
            # We can basically plot N living-score values with equal distance,
            # and divide its X-axis by K-1 to parts in equal distance to get
            #  a value for each of K items.

            n = len(bkg_livingscores)
            k = len(all_rnames)
            xp = np.arange(n)
            fp = bkg_livingscores
            x = np.linspace(0, n, k)

            ordered_rt_lvs = np.interp(x, xp, fp)
            rn2lvs = {rn: ordered_rt_lvs[sorted_rn_ranks.index(roomtype_ranking(rn))] for rn in all_rnames}

            # Apply to each room
            # ( use mean when many candidate room-types are available )
            rt_livingscores = {ri: np.mean([rn2lvs[rn] for rn in rns]) for ri, rns in amd_rtid_names.items()}

        amd_rt_livingscores[amd_id] = rt_livingscores
        bar.update(i+1)

    bar.finish()
    print len(no_amd_rtnames)
    print len(no_livingscores)

    return amd_rt_livingscores

if __name__ == '__main__':
    d = assign_amd_roomtype_livingscores()
    # with open('amadeus_roomtypes_livingscores.json', 'w') as f:
    #     json.dump(d, f)

