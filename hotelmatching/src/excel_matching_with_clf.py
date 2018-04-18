


if __name__ == '__main__':
    from excel_matches_livingscores import *

    print("Loading Amadeus hotels")
    amdh_full = pd.read_excel('./Amadeus All Properties - FEB 2017 17022017 .xlsx', header=0, index_col='PROPERTY_CODE', skiprows=1)
    amdh = convert_amd_df_to_matching_format(amdh_full)
    namdh = len(amdh)
    print("Loaded %d hotels" % namdh)

    print("Loading Booking hotels")
    bkgh = load_booking()
    # bkgh = load_booking_from_mysql()
    print("Loaded %d hotels" % len(bkgh))


    print("Building Geo Index")
    geo_index = GeoGridIndex()
    for i, hb in bkgh.iterrows():
        if hb['lat'] == 90:
            hb['lat'] = -90.0
        geo_index.add_point(GeoPoint(hb['lat'], hb['lng'], ref=hb))


    from db_matching_with_clf import *

    print("Generating match candidates")

    amdh['amd_id'] = amdh['property_code'] # we need this because extract candidates was made for Amadeus IDs
    candidates = extract_candidates(amdh, geo_index, 6, 0.7, namdh)
    mdf = pd.DataFrame(concatenate(candidates.values()))


    # with open('rdf_matches.pickle','rb') as f:
    #     clf = pickle.load(f)

    print("Using RDF classifier to pick the best matches")
    # This is the model refitted to selected matches
    with open('rdf_matches_cands.pickle','rb') as f:
        clf = pickle.load(f)

    retrain_set = []
    retrain_labels = []
    matches = {}

    for amd_id, match_cands in candidates.items():
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
            matches[d['amd_id']] = d['bkg_id']

    with open('matches_excel_rdf.json', 'w') as f:
        json.dump(matches, f)