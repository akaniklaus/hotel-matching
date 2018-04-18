import json
from collections import defaultdict, Counter
import psycopg2

def get_conn_amd():
    c = psycopg2.connect(database='d36tike0jg2ieb',
                        user='u4esungn748mp7',
                        password='pb31e2aedb696ceb6dffc2c4f0756fd59ec9838854b739fe027d7cc44a2c13fcb',
                        host='ec2-52-212-211-39.eu-west-1.compute.amazonaws.com',
                        port=5432)
    return c

with open('room_types.json') as f:
    bkg_rts = json.load(f)

with open('duplicates.json') as f:
    dups = json.load(f)

with open('similar.json') as f:
    sims = json.load(f)

def get_most_common(l):
    c = Counter(l)
    return [w for w, n in c.most_common() if n == c.most_common(1)[0][1]]

def tokenize(s):
    return [x.strip() for x in s.split()]

def rt_bkg_to_amd(rt_bkg):
    rts = set()
    rt_bkg = rt_bkg.lower()
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
        'access': 'Accessible room',
        'accessible': 'Accessible room',
        'disability': 'Accessible room',
        'presidential': 'Penthouse',
        'basic': 'Budget room',
        'classic': 'Budget room'

    }
    for kws, rt in kw_map.items():
        if all(kw in tokenize(rt_bkg) for kw in kws.split()):
            rts.add(rt)
    if 'suite' in rt_bkg and 'Concierge/Executive Suite' not in rts:
        rts.add('Junior Suite/Mini Suite')
    if not rts:
        rts.add('Standard')

    return list(rts)

def get_img_roomtypes():
    img_roomtypes = defaultdict(list)

    for imgdups in dups.values():
        for url, (durl, r) in imgdups.items():
            rt_bkg = r[6:]
            img_roomtypes[url] += rt_bkg_to_amd(rt_bkg)

    # sims
    for imgsims in sims.values():
        for url, simlist in imgsims.values():
            for (url, r), dist in simlist:
                rt_bkg = r[6:]
                img_roomtypes[url] += rt_bkg_to_amd(rt_bkg)

    with open('img_roomtypes.json', 'w') as f:
        json.dump(img_roomtypes, f)

def get_rt_from_typecode(typecode):
    d = {
        'A': 'Superior',
        'B': 'Business room',
        'C': 'Concierge/Executive Suite',
        'D': 'Deluxe Room',
        'E': 'Executive room',
        'F': 'Family Room',
        'G': 'Comfort room',
        'H': 'Accessible room',
        'I': 'Budget room',
        'L': 'Studio',
        'M': 'Standard',
        'P': 'Penthouse',
        'R': 'Residential apartment',
        'S': 'Junior Suite/Mini Suite',
        'V': 'Villa',
        'X': 'Duplex'
    }
    return d.get(typecode[0]) if typecode else None

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
        'access': 'Accessible room',
        'accessible': 'Accessible room',
        'disability': 'Accessible room',
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

AMD_TYPECODE_TO_ROOMNAME = {
    'A': 'Superior',
    'B': 'Business room',
    'C': 'Concierge/Executive Suite',
    'D': 'Deluxe Room',
    'E': 'Executive room',
    'F': 'Family Room',
    'G': 'Comfort room',
    'H': 'Accessible room',
    'I': 'Budget room',
    'L': 'Studio',
    'M': 'Standard',
    'P': 'Penthouse',
    'R': 'Residential apartment',
    'S': 'Junior Suite/Mini Suite',
    'V': 'Villa',
    'X': 'Duplex'
}

AMD_ROOMNAMES = AMD_TYPECODE_TO_ROOMNAME.values()

def extract_amd_rt_names(typecode, name, description):
    amd_rt_names = set()

    if name in AMD_ROOMNAMES:
        amd_rt_names.add(name)
    
    if typecode:
        amd_rt_name = AMD_TYPECODE_TO_ROOMNAME.get(typecode[0])
        if amd_rt_name:
            amd_rt_names.add(amd_rt_name)

    desc = description.split('\n')[0]
    desc_names = rt_description_to_amd_rtname(desc)
    if desc_names:
        amd_rt_names.update(desc_names)

    return list(amd_rt_names)

def assign_roomtypes():
    # Use img matches to generate structure
    # amd_id -> ( img -> amd_roomtype )

    # load duplicates
    with open('duplicates.json') as f:
        dups = json.load(f)

    # load similar
    with open('similar.json') as f:
        sims = json.load(f)

    # combine similar and duplicates to h_match -> (img -> [B_roomtype]) maps
    img_bkg_rts = defaultdict(dict)
    for h_match, d in dups.items():
        img_bkg_rts[h_match] = {img: [t[1][6:]] for img, t in d.items()}

    for h_match, d in sims.items():
        img_bkg_rts[h_match].update({img: [t[0][1][6:] for t in l] for img, l in d.items()})

    img_amd_rts = defaultdict(dict)
    # iterate over hotel matches
    for i, (h_match, img_brts) in enumerate(img_bkg_rts.items()):
        if i % 10 == 0:
            prog = i * 100.0 / len(img_bkg_rts)
            print "%.2f %%" % prog
        amd_id, bkgid = h_match.split()

        # Fetch Amadeus room types
        conn_amd = get_conn_amd()
        cur = conn_amd.cursor()

        amd_rts = set()
        query = """
                    SELECT "typeCode", name, td.text
                    FROM public."RoomTypes" rt
                    JOIN public."TextDescriptions" td ON td.relation = rt.id
                    WHERE property = '%s'
                """ % amd_id

        cur.execute(query)
        for typecode, name, description in cur.fetchall():
            rts = extract_amd_rt_names(typecode, name, description)
            if rts:
                amd_rts.update(rts)

        # find best Amadeus room type for each B.com room type
        # and assign to image
        for img, bkg_rts in img_brts.items():
            img_rts = []
            for rt_bkg in bkg_rts:
                img_rts += [rt for rt in rt_bkg_to_amd(rt_bkg) if rt in amd_rts]

            # keep majority room types when many options available
            if img_rts:
                img_amd_rts[amd_id][img] = get_most_common(img_rts)

    return img_amd_rts

def get_rt_map():
    rt_map = {}
    for bkg_rt in bkg_rts:
        rt_map[bkg_rt] = rt_bkg_to_amd(bkg_rt)

    return rt_map

if __name__ == '__main__':
    # get_img_roomtypes()
    # rt_map = get_rt_map()
    # with open('bkg_rt_to_amd.json', 'w') as f:
    #     json.dump(rt_map, f, sort_keys=True)

    d = assign_roomtypes()
    with open('amadeus_images_roomtype_assignments.json', 'w') as f:
        json.dump(d, f)

    with open('amadeus_images_roomtype_assignments.json') as f:
        d = json.load(f)        
