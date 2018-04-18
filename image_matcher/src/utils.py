from __future__ import print_function
from os.path import exists
from os import makedirs
import time
import json

def create_dir_if_needed(path):
    if not exists(path):
        makedirs(path)
        return True
    else:
        return False

def format_time(t):
    return time.strftime('%Hh:%Mm:%Ss', time.gmtime(t))

def timeit(msg):
    def _timeit(f):
        def timed(*args, **kw):
            print("=" * len(msg))
            print(msg)
            ts = time.time()
            result = f(*args, **kw)
            te = time.time()
            elapsed = te - ts
            print('Took %s' % format_time(elapsed))
            return result
        return timed
    return _timeit

def load_matches():
    with open('/mnt/data/projects/roomimages/matches.json') as f:
        matches = json.load(f)
    return matches