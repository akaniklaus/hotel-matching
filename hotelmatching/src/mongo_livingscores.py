#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pymongo
import pymysql
import pandas as pd
import numpy as np


MONGODB_SERVER = "52.233.173.236"
MONGODB_PORT = 27017
MONGODB_USER = 'livingrooms:TXrW1IhGzp'

HOTEL_TYPE_MAPS = {
    None: 'OTHER',
    u'': 'OTHER',
    u'apartments': 'APARTMENTS',
    u'bed and breakfast': 'BED AND BREAKFAST',
    u'boat': 'OTHER',
    u'country house': 'COUNTRY HOUSE',
    u'farm': 'OTHER',
    u'guest house': 'GUEST HOUSE',
    u'home': 'HOME',
    u'hostal': 'HOSTAL',
    u'hostel': 'HOSTEL',
    u'hotel': 'HOTEL',
    u'inn': 'INN',
    u'lodge': 'LODGE',
    u'motel': 'MOTEL',
    u'studio': 'APARTMENTS',
    u'tent': 'OTHER',
    u'villa': 'VILLA'
}


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

def fetch_city_data():
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

    query = """SELECT c.CityID, c.CityName, r.CountryCode from
                tbl_Cities c join tbl_Regions r on c.regionid=r.RegionID
            """

    cur.execute(query)

    cities_data = {}
    for row in cur:
        cid, name, ccode = row
        cities_data[cid] = {
            'name': name,
            'ccode': ccode
        }

    cur.close()
    conn.close()

    return cities_data


if __name__ == '__main__':
    rows_lvs = []
    rows_nolvs = []
    c = get_conn_bkg()

    cities_data = fetch_city_data()
    
    for h in c.find({}):
        row = {
            'HotelName': h['title'],
            'HotelType': HOTEL_TYPE_MAPS[h['type']]
        }
        
        # fetch Country, City from MySQL
        cdata = cities_data.get(int(h['city_id']))
        if not cdata:
            continue
        row['City'] = cdata['name'] if cdata else ''
        row['Country'] = cdata['ccode'] if cdata else ''

        # MinScore, AvgScore, MaxScore
        lvs_rooms = [r for r in h['rooms'] if 'livingscore' in r]
        lvscores = np.array([r['livingscore'] for r in lvs_rooms])

        if len(lvscores):
            row['MinScore'] = np.min(lvscores)
            row['MaxScore'] = np.max(lvscores)
            row['AvgScore'] = np.mean(lvscores)

            rows_lvs.append(row)
        else:
            rows_nolvs.append(row)

    columns = 'HotelName, HotelType, City, Country, MinScore, AvgScore, MaxScore'.split(', ')
    df = pd.DataFrame(data=rows_lvs, columns=columns)
    df.to_excel('bkglvs.xlsx', index=False)

    columns = 'HotelName, HotelType, City, Country'.split(', ')
    df = pd.DataFrame(data=rows_nolvs, columns=columns)
    df.to_excel('bkgnolvs.xlsx', index=False)
