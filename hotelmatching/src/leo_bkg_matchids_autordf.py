# coding: utf-8
from leo_bkg_matchids import *
import itertools
from scipy.sparse import hstack


def concatenate(ll):
    return list(itertools.chain.from_iterable(ll))

# Feature extractors & transformers
def norm_text(column):
    return column.astype(unicode).fillna('').apply(normalize).astype(unicode)

def row_name_included(row):
    n = normalize(row['name']).replace(' ', '')    
    nb = normalize(row['name_bkg']).replace(' ', '')
    if not n:
        return False
    if not nb:
        return False
    return (n in nb) or (nb in n)    

def row_chain_included(row):
    n = normalize(row['chain']).replace(' ', '')    
    nb = normalize(row['chain_bkg']).replace(' ', '')
    if not n:
        return False
    if not nb:
        return False
    return (n in nb) or (nb in n)

def row_chain_sim(row):
    return get_name_sim(row['chain'], row['chain_bkg'])

def row_chain_sim_sw(row):
    return get_name_sim(row['chain'], row['chain_bkg'], swap_words=True)

def load_chain_encoders():
    with open('chain_encoders.pickle', 'rb') as f:
        le_chain, ohe, le_chain_bk, ohe_bk = pickle.load(f)
    return le_chain, ohe, le_chain_bk, ohe_bk

def add_features(X):
    for col in 'name name_bkg chain chain_bkg'.split():
        X[col] = norm_text(X[col])
    
    X['name_included'] = X.apply(row_name_included, axis=1)
    X['chain_included'] = X.apply(row_chain_included, axis=1)

    #     X['chain_sim'] = X.apply(row_chain_sim, axis=1)
    #     X['chain_sim_sw'] = X.apply(row_chain_sim_sw, axis=1)
    # No chain names on this dataset
    X['chain_sim'] = 0.0
    X['chain_sim_sw'] = 0.0
        
    return X

def pre_process(X):    
    X.chain.fillna('', inplace=True)
    X.chain_bkg.fillna('', inplace=True)
    X.name.fillna('', inplace=True)
    X.name_bkg.fillna('', inplace=True)
    X.chain_included.fillna(False, inplace=True)
    X.name_included.fillna(False, inplace=True)

    Xnum = X[[u'dist', u'name_sim', u'name_sim_sw', u'chain_sim', u'chain_sim_sw',
           u'name_included', u'chain_included']]

    Xcat = X[[u'chain', u'chain_bkg']]

    le_chain, ohe, le_chain_bk, ohe_bk = load_chain_encoders()
    Xchain = le_chain.transform(X.chain)
    Xchain = ohe.transform(Xchain.reshape(-1,1))

    Xchain_bkg = le_chain_bk.transform(X.chain_bkg)
    Xchain_bkg = ohe_bk.transform(Xchain_bkg.reshape(-1,1))

    X = hstack((Xnum.astype(float), Xchain, Xchain_bkg))
    
    return X

if __name__ == '__main__':
    # # 1. Load Leonardo & Booking hotels
    print("1. Load Leonardo & Booking hotels")
    leoh, bkgh, matches, geo_index, nleoh = initialize_matching(True)


    # # 2. Generate candidates
    print("2. Generate candidates")
    candidates = extract_candidates(leoh, geo_index, 1, 0.7, nleoh)


    # # 3. Extract features and classify good/bad matches
    print("3. Extract features and classify good/bad matches")

    mdf = pd.DataFrame(concatenate(candidates.values()))
    with open('rdf_matches.pickle','rb') as f:
        clf = pickle.load(f)

    retrain_set = []
    retrain_labels = []
    matches = {}
    for leo_id, match_cands in candidates.items():
        # filter candidates with classifier
        if not match_cands:
            continue
        X = pd.DataFrame(match_cands)
        X = add_features(X)
        X['match'] = clf.predict(pre_process(X))
        X = X[X.match == 1]
        
        ncands = X.shape[0]
        
        if ncands == 0:
            match = None
        elif ncands == 1:
            match = X.iloc[0,:]
        else:        
            # pick best if many different
            sums = {}
            compare_cols = 'name_sim_sw name_included chain_included chain_sim_sw'.split()
            for c in compare_cols:
                sums[c] = X[c].sum()
            sums['inv_dist'] = sum(1/d for d in X.dist.notnull())
            
            def score_row(row):
                score = sum(row[c]/sums[c] for c in compare_cols if sums[c])
                if row['dist']:
                    score += (1 / row['dist']) / sums['inv_dist']
                else:
                    score += 1
                return score
            
            X.score = X.apply(score_row, axis=1)
            ind_max = X.score.argmax()
            match = X.loc[ind_max,:]

            # mark the 'losers' as non-matches for future re-training
            for i in X.index:
                if i != ind_max:
                    d = X.loc[i,:].to_dict()
                    retrain_set.append(d)
                    retrain_labels.append(0)
        
        if match is not None:
            d = match.to_dict()
            retrain_set.append(d)
            retrain_labels.append(1)
            matches[d['leo_id']] = d['bkg_id']


    # # 4. filter matches with selected ids
    print("4. Filter matches with selected ids")
    with open('bkgids.txt') as f:
        bkgids = [int(l.strip()) for l in f.readlines()]

    filtered_matches = {lid: bid for (lid, bid) in matches.items() if bid in bkgids}

    match_details = [d for i, d in enumerate(retrain_set) \
                         if retrain_labels[i] and\
                         d['bkg_id'] in bkgids]
    detail_fields = 'bkg_id leo_id name name_bkg dist name_sim name_sim_sw'.split()
    match_details = [{k: d[k] for k in detail_fields} for d in match_details]

    with open('match_details.json', 'w') as f:
        json.dump(match_details, f)

    perc_matched = len(matches) * 100.0 / nleoh
    print("%.1f%% Leonardo properties matched to some B.com property" % perc_matched)

    perc_matched = len(filtered_matches) * 100.0 / nleoh
    print("%.1f%% matched to a top Livinscore B.com property" % perc_matched)
