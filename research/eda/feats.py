from load import *
G = games.set_index('game_id')
trd = tr.drop_duplicates().copy()
trd['n'] = trd.groupby('session_id').session_id.transform('size')
trd['blank_ps'] = trd.play_seconds.isna() & (trd.play_now == 1)
fr = pd.concat([friends, friends.rename(columns={'user_id': 'friend_id', 'friend_id': 'user_id'})])
fmap = fr.groupby('user_id').friend_id.apply(set).to_dict()
HCOLS = ['user_id', 'session_id', 'ts', 'game_id', 'rel', 'play_now', 'play_seconds', 'completed', 'score_percentile', 'watch_ms']


def gattr(df):
    for c in ['mechanic', 'family', 'theme', 'art_style', 'pace', 'difficulty', 'typical_session_sec',
              'has_leaderboard', 'editorial_stars', 'added_day']:
        df[c] = df.game_id.map(G[c])
    return df


def build(T, H):
    """T: target rows. H: history impressions. Uses strictly earlier ts only (a session's rows share one ts)."""
    H = gattr(H.copy())
    T = gattr(T.copy())
    H['played'] = (H.play_now == 1).astype(int)
    H['quit'] = ((H.play_now == 1) & (H.play_seconds < 20)).astype(int)
    H['good'] = (H.rel >= 1).astype(int)
    H['comp'] = (H.rel >= 2).astype(int)
    Hg = {u: g.sort_values('ts') for u, g in H.groupby('user_id')}
    names = ['u_n', 'u_rate', 'u_play', 'u_skill', 'u_skill_n', 'u_comp', 'u_quitrate', 'mech_n', 'mech_good',
             'fam_n', 'fam_good', 'seen_n', 'played_n', 'best_prev', 'last_prev', 'seen_days_ago', 'theme_n',
             'theme_good', 'art_n', 'art_good', 'pace_n', 'pace_good', 'u_watch_mean', 'u_sessions']
    out = {k: np.full(len(T), np.nan) for k in names}
    Tr = T.reset_index(drop=True)
    ts_arr = Tr.ts.values
    gid_arr = Tr.game_id.values
    catcols = [('mechanic', 'mech'), ('family', 'fam'), ('theme', 'theme'), ('art_style', 'art'), ('pace', 'pace')]
    for u, idx in Tr.groupby('user_id').groups.items():
        h = Hg.get(u)
        if h is None or len(h) == 0:
            for k in ['u_n', 'seen_n', 'played_n', 'mech_n', 'fam_n', 'mech_good', 'fam_good', 'theme_n',
                      'theme_good', 'art_n', 'art_good', 'pace_n', 'pace_good', 'u_sessions']:
                out[k][idx] = 0
            continue
        hts = h.ts.values
        good = h.good.values
        played = h.played.values
        comp = h.comp.values
        quit_ = h.quit.values
        cs = lambda a: np.concatenate([[0], np.cumsum(a)])
        cg, cp, cc, cq = cs(good), cs(played), cs(comp), cs(quit_)
        sp = h.score_percentile.values
        spm = ~np.isnan(sp)
        csp, csn = cs(np.where(spm, sp, 0)), cs(spm)
        wv = h.watch_ms.values
        wm = ~np.isnan(wv)
        cw, cwn = cs(np.where(wm, wv, 0)), cs(wm)
        sess = h.session_id.values
        first = np.concatenate([[1], (sess[1:] != sess[:-1]).astype(int)])
        csess = cs(first)
        cat = {}
        for c, pre in catcols:
            vals = h[c].values
            cat[c] = {v: (cs(vals == v), cs((vals == v) & (good == 1))) for v in set(vals)}
        gids = h.game_id.values
        rels = h.rel.values
        for i in idx:
            t = ts_arr[i]
            k = np.searchsorted(hts, t, side='left')
            out['u_n'][i] = k
            out['u_sessions'][i] = csess[k]
            if k > 0:
                out['u_rate'][i] = cg[k] / k
                out['u_play'][i] = cp[k] / k
                out['u_comp'][i] = cc[k] / k
                out['u_quitrate'][i] = cq[k] / cp[k] if cp[k] > 0 else np.nan
                if csn[k] > 0:
                    out['u_skill'][i] = csp[k] / csn[k]
                    out['u_skill_n'][i] = csn[k]
                if cwn[k] > 0:
                    out['u_watch_mean'][i] = cw[k] / cwn[k]
            for c, pre in catcols:
                v = Tr.at[i, c]
                d = cat[c]
                if v in d:
                    out[pre + '_n'][i] = d[v][0][k]
                    out[pre + '_good'][i] = d[v][1][k]
                else:
                    out[pre + '_n'][i] = 0
                    out[pre + '_good'][i] = 0
            gm = np.where(gids[:k] == gid_arr[i])[0]
            out['seen_n'][i] = len(gm)
            if len(gm):
                out['played_n'][i] = played[gm].sum()
                out['best_prev'][i] = rels[gm].max()
                out['last_prev'][i] = rels[gm[-1]]
                out['seen_days_ago'][i] = (t - hts[gm[-1]]) / 86400
            else:
                out['played_n'][i] = 0
    for k, v in out.items():
        Tr[k] = v
    return Tr


def add_friends(T, H):
    Hg = H[H.rel >= 1].groupby('game_id')
    byg = {g: d[['user_id', 'ts', 'rel']].values for g, d in Hg}
    f1 = []
    f2 = []
    for u, g, t in zip(T.user_id, T.game_id, T.ts):
        fs = fmap.get(u, set())
        a = byg.get(g)
        n1 = n2 = 0
        if a is not None and fs:
            for uu, tt, r in a:
                if tt < t and uu in fs:
                    n1 += 1
                    n2 += r >= 2
        f1.append(n1)
        f2.append(n2)
    T['fr_good'] = f1
    T['fr_comp'] = f2
    T['n_friends'] = T.user_id.map(lambda u: len(fmap.get(u, ())))
    return T


def prefix_feats(T, P):
    P = gattr(P.copy())
    P['quit'] = ((P.play_now == 1) & (P.play_seconds < 20)).astype(int)
    P['good'] = (P.rel >= 1).astype(int)
    P['comp'] = (P.rel >= 2).astype(int)
    P['played'] = P.play_now
    g = P.groupby('session_id')
    base = pd.DataFrame({'p_good': g.good.sum(), 'p_comp': g.comp.sum(), 'p_quit': g.quit.sum(),
                         'p_played': g.played.sum(), 'p_watch': g.watch_ms.mean(), 'p_relmean': g.rel.mean()})
    T = T.merge(base, left_on='session_id', right_index=True, how='left')
    tabs = {sid: d for sid, d in P.groupby('session_id')}
    for c in ['family', 'mechanic']:
        s_good, s_quit, s_n, s_rel, s_comp = [], [], [], [], []
        for sid, v in zip(T.session_id, T[c]):
            d = tabs[sid]
            m = d[d[c] == v]
            s_n.append(len(m))
            s_good.append(m.good.sum())
            s_quit.append(m.quit.sum())
            s_comp.append(m.comp.sum())
            s_rel.append(m.rel.mean() if len(m) else np.nan)
        T['ps_' + c + '_n'] = s_n
        T['ps_' + c + '_good'] = s_good
        T['ps_' + c + '_quit'] = s_quit
        T['ps_' + c + '_comp'] = s_comp
        T['ps_' + c + '_rel'] = s_rel
    return T


if __name__ == '__main__':
    Tt = trd[(trd.n >= 7) & (trd.position > 3) & (~trd.blank_ps)].copy()
    Tt = build(Tt, trd[HCOLS])
    Tt = add_friends(Tt, trd)
    P = trd[(trd.position <= 3) & (trd.n >= 7)]
    Tt = prefix_feats(Tt, P)
    Tt.to_pickle('train_feats.pkl')
    V = vs.merge(vl, on=['session_id', 'game_id'])
    V['ts'] = V.ts.astype(float)
    V = build(V, trd[HCOLS])
    V = add_friends(V, trd)
    V = prefix_feats(V, vp)
    V['day'] = (V.ts // 86400).astype(int)
    V = V.rename(columns={'relevance': 'rel'})
    V.to_pickle('val_feats.pkl')
    print(len(Tt), len(V))
