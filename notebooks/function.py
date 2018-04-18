import pymongo 
import re 
import random 



def clean_record(json_record):
    record = json.loads(json_record)
    return pandas.io.json.json_normalize(record)

def flatten_record(json_record):
    record = json.loads(json_record)
    facilities = pandas.io.json.json_normalize(record['facilities'])
    number_of_rooms = record['number_of_rooms']
    return facilities

def json_flatten(json_record, sep='::'):

    def flatten(record, node=''):
        new_record = {}
        if type(record) is dict:
            for k, v in record.items():
                if node is not '':
                    new_record.update(flatten(v, node + sep + k))
                else:
                    new_record.update(flatten(v, node + k)) 
        else:
            new_record.update({node : record })
        return new_record

    record = flatten(json_record)

    return record


def remove_empty_nodes(record):
    new_record = record.copy()
    for k, v in record.items():
        if type(v) is list:
            if len(v) is 0:
                del new_record[k]
        elif type(v) is str:
            if len(v.strip()) is 0:
                del new_record[k]
    return new_record

def remove_nodes(record, nodes=[]):
    new_record = record.copy()
    for node in nodes:
        if node in new_record:
            del new_record[node]
    return new_record

def list_to_features(list_feature, prefix=''): 
    if type(list_feature) is list:
        return {prefix + value.strip(): 1 for value in list_feature} 
    
def string_to_features(string_feature, prefix='', sep=','):
    if type(string_feature) is str:
        return {prefix + value.strip(): 1 for value in string_feature.split(sep) if len(value.strip()) is not 0}
    
def normalize_features(features): 
    return ['_'+re.sub('[^a-zA-Z0-9_.]', '', feature.strip().replace(" ", "_").replace("/", "_").replace("-", "_").replace(">", "greater").replace("<", "lower")) for feature in features]

def rooms_features(record): 
    new_records = []
    for room in record['rooms']: 
        facilities = room['facilities']
        size_string = re.findall('\d*\.?\d*[ ]* m²', facilities)
        if len(size_string) > 0: 
            size = float(re.findall('\d*\.?\d*', size_string[0])[0])
        else: 
            size = 0 
        facilities = re.sub('\d*\.?\d*[ ]* m²', '', facilities)
        new_record = string_to_features(facilities) 
        #if size > 0: 
        #    new_record.update({'Size': size})
        name = room['name']
        new_records.append(new_record) 
    return new_records 

def extract_size(feature):  
    size_string = re.findall('\d*\.?\d*[ ]* m²', feature)
    return float(re.findall('\d*\.?\d*', size_string[0])[0]) if len(size_string) > 0 else 0.0     
    
def extract_view(feature): 
    return re.sub('view', '', feature).split('/') if 'view' in feature else []  

def extract_internet(facility): 
    internet = {}
    for value in facility: 
        if 'Free' in value: 
            internet.update({'Free_Internet': 1})
        if 'WiFi' in value or 'Wireless' in value: 
            internet.update({'WiFi': 1})
        if 'and costs' in value or 'paid' in value: 
            internet.update({'Paid_Internet': 1})
    return internet 
                
def extract_parking(facility): 
    parking = {}
    for value in facility: 
        if 'Free' in value: 
            parking.update({'Free_Parking': 1})
        if 'Private parking is possible' in value: 
            parking.update({'Private_Parking': 1})
        if 'Public parking is possible' in value: 
            parking.update({'Public_Parking': 1})
        if 'reservation is not needed' in value: 
            parking.update({'Parking_Reservation_Not_Needed': 1})
        if 'reservation is needed' in value: 
            parking.update({'Parking_Reservation_Needed': 1})
        if 'reservation is not possible' in value: 
            parking.update({'Parking_Reservation_Not_Possible': 1})
        if 'and costs' in value: 
            parking.update({'Paid_Parking': 1}) 
    return parking 
            
def hotel_facilities_to_features(hotel_facilities): 
    del hotel_facilities['languages_spoken'] 
    new_record = {} 
    new_record.update(extract_internet(hotel_facilities['internet'])) 
    del hotel_facilities['internet']
    new_record.update(extract_internet(hotel_facilities['parking'])) 
    del hotel_facilities['parking']
    for key, facility in hotel_facilities.items(): 
        new_record.update(list_to_features(facility))
    return new_record 
    
def room_facilities_to_features(room_facilities): 
    new_record = {} 
    features = room_facilities.split(',')  #list of strings 
    for feature in features: 
        size = extract_size(feature) 
        if size > 0: 
            new_record.update({'Size': size}) 
        else: 
            view = extract_view(feature)  
            if len(view) > 0: 
                new_record.update(list_to_features(view, 'View_')) 
            else: 
                new_record.update({feature: 1})  
    return new_record 

def hotel_features(record): 
    new_record = {} 
    new_record.update(hotel_facilities_to_features(record['facilities'])) 
    #if 'number_of_rooms' in record: 
    #    new_record.update({'number_of_rooms': int(record['number_of_rooms'])})
    #if len(record['surroundings']) > 0: 
    #    new_record.update(string_to_features(record['surroundings']))
    return new_record 

def rooms_features(record): 
    new_records = []
    for room in record['rooms']: 
        if 'facilities' in room: 
            new_record = room_facilities_to_features(room['facilities'])
            name = room['name']
            new_records.append(new_record) 
    return new_records 


def prepare_record(record):
    new_record = record.copy()
    for v in new_record['facilities']:
        new_record['facilities'][v] = list_to_features(record['facilities'][v])
    if len(record['surroundings'].strip()) is not 0:
        new_record['surroundings'] = string_to_features(record['surroundings'])
    else:
        del new_record['surroundings'] 
    new_record = remove_nodes(new_record, ['_id', 'rooms', 'description', 'reviews', 'url', 'updated', 'chain', 'popular_facilities', 'title', 'highlights', 'check_out', 'check_in', 'hotel_id'])
    new_record = json_flatten(new_record, '.')
    new_record = remove_empty_nodes(new_record)
    if 'number_of_rooms' in new_record: 
        #new_record['number_of_rooms'] = int(new_record['number_of_rooms']) 
        del new_record['number_of_rooms'] 
    return { re.sub('[^a-zA-Z0-9_.]', '', k.replace(" ", "_").replace("/", "_").replace("-", "_").replace(">", "greater").replace("<", "lower")): v for k, v in new_record.items()}

def mongo_collection(host='localhost', port=27017, db='booking', collection='hotel_details'):
    client = pymongo.MongoClient(host, port)
    return client[db][collection]

def hotel_records(records): 
    new_records = []
    y = []
    for record in records:
        new_record = hotel_features(record)  
        new_records.append(new_record)
        y.append(record['type'])  
    return new_records, y 
    
def rooms_records(records): 
    new_records = []
    y = []
    for record in records:
        for new_record in rooms_features(record): 
            new_records.append(new_record)
            y.append(record['type'])  
    return new_records, y 

def split_records(records, ratio): 
    train = [] 
    valid = []
    for record in records: 
        if 'reviews' in record: 
            del record['reviews'] 
        if random.random() <= ratio: 
            train.append(record) 
        else: 
            valid.append(record)    
    return train, valid 