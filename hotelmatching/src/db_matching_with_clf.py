# coding: utf-8

from amd_bkg_matchids import *
from scipy.sparse import hstack
import itertools


# # 4. extract features
def concatenate(ll):
    return list(itertools.chain.from_iterable(ll))

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

# # 5. use classifier to pick valid matches
def load_chain_encoders():
    with open('chain_encoders.pickle', 'rb') as f:
        le_chain, ohe, le_chain_bk, ohe_bk = pickle.load(f)
    return le_chain, ohe, le_chain_bk, ohe_bk

def add_features(X):
    for col in 'name name_bkg chain chain_bkg'.split():
        X[col] = norm_text(X[col])
    
    X['name_included'] = X.apply(row_name_included, axis=1)
    X['chain_included'] = X.apply(row_chain_included, axis=1)

    X['chain_sim'] = X.apply(row_chain_sim, axis=1)
    X['chain_sim_sw'] = X.apply(row_chain_sim_sw, axis=1)
    
    return X

def pre_process_cand(X):    
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

    # In[152]:
    # # 1. load Amadeus & Booking hotels
    amdh, bkgh, matches, geo_index, namdh = initialize_matching(True)


    # # 3. generate candidates
    candidates = extract_candidates(amdh, geo_index, 6, 0.7, namdh)
    mdf = pd.DataFrame(concatenate(candidates.values()))


    # with open('rdf_matches.pickle','rb') as f:
    #     clf = pickle.load(f)

    # This is the model refitted to selected matches
    with open('rdf_matches_cands.pickle','rb') as f:
        clf = pickle.load(f)

    retrain_set = []
    retrain_labels = []
    matches = {}

    for lvr_id, match_cands in candidates.items():
        # filter candidates with classifier
        if not match_cands:
            continue
        X = pd.DataFrame(match_cands)
        X = add_features(X)
        X['match'] = clf.predict(pre_process_cand(X))
        X = X[(X.match == 1) & ~X.bkg_id.isin(matches.values())]
        
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
            matches[d['lvr_id']] = d['bkg_id']

