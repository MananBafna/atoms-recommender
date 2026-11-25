from load import *
import math
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 50)
Tt = pd.read_pickle('train_feats2.pkl')
V = pd.read_pickle('val_feats2.pkl')
trd = tr.drop_duplicates().copy()
G = games.set_index('game_id')


def pair_auc(D, col, sign=1):
    num = den = 0.0
    x = D[['session_id', 'rel', col]].dropna()
    for sid, d in x.groupby('session_id'):
        r = d.rel.values
        s = d[col].values * sign
        if len(d) < 2:
            continue
        dr = r[:, None] - r[None, :]
        ds = s[:, None] - s[None, :]
        m = dr > 0
        den += m.sum()
        num += ((ds > 0) & m).sum() + 0.5 * ((ds == 0) & m).sum()
    return num / den if den else np.nan


def ndcg(D, col, k=5, sign=1):
    res = []
    for sid, d in D.groupby('session_id'):
        gains = (2 ** d.rel.values - 1)
        if gains.sum() == 0:
            continue
        order = np.argsort(-(d[col].fillna(d[col].mean()).values * sign), kind='stable')
        dcg = sum(gains[o] / math.log2(i + 2) for i, o in enumerate(order[:k]))
        ideal = sorted(gains, reverse=True)[:k]
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        res.append(dcg / idcg)
    return np.mean(res), len(res)


# 1. game mean watch (excluding blank and exact-0) from train
w = trd[(trd.watch_ms > 0)].groupby('game_id').watch_ms.agg(['mean', 'size'])
glob_w = trd[trd.watch_ms > 0].watch_ms.mean()
w['sm'] = (w['mean'] * w['size'] + 30 * glob_w) / (w['size'] + 30)
for D in (Tt, V):
    D['gwatch'] = D.game_id.map(w.sm)
print('=== game mean watch (excl 0/blank), smoothed ===')
print('within-session pairwise concordance: train', round(pair_auc(Tt, 'gwatch'), 4), ' val', round(pair_auc(V, 'gwatch'), 4), ' (pop: val 0.574)')
print('NDCG@5 val by pop', ndcg(V, 'pop'), ' gwatch', ndcg(V, 'gwatch'))
# game-level correlation watch vs pop
gg = trd.groupby('game_id').agg(rate=('rel', lambda s: (s >= 1).mean()), n=('rel', 'size')).join(w)
print('spearman(game mean watch, P(r>=1)) all games:', round(gg['mean'].corr(gg.rate, method='spearman'), 3))
# user-game level: has watched game before -> past watch on this game
# 2. watch zero issue
print('watch==0 share', (trd.watch_ms == 0).mean().round(4), 'blank', trd.watch_ms.isna().mean().round(4))
print('watch==0 by position', trd.groupby(pd.cut(trd.position, [0, 1, 3, 10, 50]), observed=True).apply(lambda d: (d.watch_ms == 0).mean()).round(4).to_dict())
print('watch==0 by platform', trd.assign(p=trd.user_id.map(users.set_index('user_id').platform)).groupby('p').apply(lambda d: (d.watch_ms == 0).mean()).round(4).to_dict())
print('blank watch by platform', trd.assign(p=trd.user_id.map(users.set_index('user_id').platform)).groupby('p').apply(lambda d: d.watch_ms.isna().mean()).round(4).to_dict())
print('blank watch by day', trd.groupby('day').apply(lambda d: d.watch_ms.isna().mean()).round(3).to_dict())
print('blank ps (tapped) by platform', trd[trd.play_now == 1].assign(p=lambda d: d.user_id.map(users.set_index('user_id').platform)).groupby('p').apply(lambda d: d.play_seconds.isna().mean()).round(4).to_dict())

# 3. affinity among game-new-to-user
print('\n=== affinity among games the user has NOT seen before (seen_n==0) ===')
for nm, D in (('train', Tt), ('val', V)):
    d = D[D.seen_n == 0]
    print(nm, len(d), {c: round(pair_auc(d, c), 4) for c in ['aff_mech', 'aff_fam', 'aff_pace', 'pop', 'fr_good']})

# 4. friends effect controlling for popularity
print('\n=== friends effect within popularity terciles (P(r>=1), val) ===')
for nm, D in (('train', Tt), ('val', V)):
    D = D.copy()
    D['pt'] = pd.qcut(D['pop'], 3, labels=['lowpop', 'midpop', 'highpop'])
    D['fr'] = np.where(D.fr_comp > 0, 'friend_completed', 'none')
    print(nm); print(D.groupby(['pt', 'fr'], observed=True).rel.agg(['size', 'mean', lambda s: (s >= 1).mean(), lambda s: (s >= 2).mean()]).round(4))
# friend effect among users with many friends
print('val: P(r>=1) by n_friends bucket, with friend completed');
V['nfb'] = pd.cut(V.n_friends, [0, 3, 6, 10, 30])
print(V.groupby(['nfb', V.fr_comp > 0], observed=True).rel.agg(['size', lambda s: (s >= 1).mean(), 'mean']).round(4))
# friend recency: not computed; friends share of slate
print('share of slate rows with friend_good>0 val', (V.fr_good > 0).mean().round(3), 'fr_comp>0', (V.fr_comp > 0).mean().round(3), ' train', (Tt.fr_good > 0).mean().round(3))

# 5. skill x difficulty : AUC of -difficulty inside skill terciles
print('\n=== skill x difficulty: concordance of difficulty (sign -1 => easier better) by skill tercile ===')
for nm, D in (('train', Tt), ('val', V)):
    d = D[(D.u_skill_n >= 2)].copy()
    d['st'] = pd.qcut(d.u_skill, 3, labels=['low', 'mid', 'high'])
    print(nm, {str(k): round(pair_auc(g, 'difficulty', -1), 4) for k, g in d.groupby('st', observed=True)}, ' (0.5=no effect)')
    d2 = D[D.u_skill_n.isna() | (D.u_skill_n < 2)]
    print(nm, 'no skill known:', round(pair_auc(d2, 'difficulty', -1), 4), len(d2))
# skill alone: effect on level (user-level)
print('mean u_skill vs overall rel by user skill tercile (train)')
d = Tt[(Tt.u_skill_n >= 2)].copy()
d['st'] = pd.qcut(d.u_skill, 3, labels=['low', 'mid', 'high'])
print(d.groupby('st', observed=True).rel.agg(['size', 'mean', lambda s: (s >= 1).mean(), lambda s: (s >= 2).mean()]).round(4))
# quit rate as function of (difficulty-skill) in train
d = trd.merge(games, on='game_id')
d = d[d.play_now == 1]
d['quit'] = (d.play_seconds < 20).astype(int)
us = trd[trd.score_percentile.notna()].groupby('user_id').score_percentile.mean()
d['skill'] = d.user_id.map(us)
d = d.dropna(subset=['skill'])
d['st'] = pd.qcut(d.skill, 4, labels=['q1', 'q2', 'q3', 'q4'])
print('\nquit|play by user skill quartile (full-train skill, in-sample) x difficulty')
print(d.groupby(['st', 'difficulty'], observed=True).quit.mean().unstack().round(3))
print('completed|play>=20 by skill quartile x difficulty')
d2 = d[d.play_seconds >= 20]
print(d2.groupby(['st', 'difficulty'], observed=True).completed.mean().unstack().round(3))
print('score_percentile by skill q x difficulty (completed lb games)')
print(d2[d2.score_percentile.notna()].groupby(['st', 'difficulty'], observed=True).score_percentile.mean().unstack().round(1))
print('user mean skill dist', us.describe().round(1).to_dict())

# 6. cold users & new games (val)
print('\n=== COLD USERS (val): signals ===')
cold = V[V.u_sessions <= 2]
warm = V[V.u_sessions > 2]
print('cold rows', len(cold), 'sessions', cold.session_id.nunique(), ' warm rows', len(warm))
for c in ['pop', 'aff_fam', 'aff_mech', 'fr_good', 'difficulty', 'gwatch', 'ps_family_good']:
    sg = -1 if c == 'difficulty' else 1
    print(c, 'cold', round(pair_auc(cold, c, sg), 4), 'warm', round(pair_auc(warm, c, sg), 4))
print('NDCG@5 pop cold', ndcg(cold, 'pop'), 'warm', ndcg(warm, 'pop'))
print('rel dist cold', cold.rel.value_counts(normalize=True).sort_index().round(3).to_dict())

# 7. exposure vs personal taste: does logging policy personalise?
print('\n=== LOGGING POLICY: does exposure follow user taste? ===')
t = trd.merge(games[['game_id', 'mechanic', 'family']], on='game_id')
res = []
for c in ['mechanic', 'family']:
    big = t.groupby('user_id').size()
    big = big[big >= 150].index
    x = t[t.user_id.isin(big)]
    e = x.groupby(['user_id', c]).size().unstack(fill_value=0)
    e = e.div(e.sum(1), axis=0)
    r = x.groupby(['user_id', c]).rel.apply(lambda s: (s >= 1).mean()).unstack()
    # residualise by column means
    e2 = e - e.mean()
    r2 = r - r.mean()
    mask = r2.notna()
    a = e2.values[mask.values]
    b = r2.values[mask.values]
    print(c, 'users', len(big), 'corr(user exposure-share deviation, user play-rate deviation) =', round(np.corrcoef(a, b)[0, 1], 3), ' spearman', round(pd.Series(a).corr(pd.Series(b), method='spearman'), 3))
# exposure of slate game vs popularity in val: share of slate rows vs rate
print('corr over games: val slate frequency vs train rate', round(V.game_id.value_counts().reindex(games.game_id).fillna(0).corr(trd.groupby('game_id').rel.apply(lambda s: (s >= 1).mean()).reindex(games.game_id), method='spearman'), 3))
# does slate contain favourite games? share of slate rows with best_prev>=2
print('val slate rows where the user previously completed/replayed that game:', ((V.best_prev >= 2)).sum(), round((V.best_prev >= 2).mean(), 3))
# same game repeated across sessions at what lag; user exposure to a game is it concentrated?
cnt = trd.groupby(['user_id', 'game_id']).size()
print('train: user-game pairs', len(cnt), 'share seen >1:', round((cnt > 1).mean(), 3), ' mean exposures per pair', round(cnt.mean(), 2))
# hist: does the earlier feed repeat games the user liked? P(shown again | prev rel)
t2 = trd.sort_values(['user_id', 'ts'])
t2['k'] = t2.groupby(['user_id', 'game_id']).cumcount()
first = t2[t2.k == 0][['user_id', 'game_id', 'rel']].rename(columns={'rel': 'rel1'})
second = t2[t2.k == 1][['user_id', 'game_id']].assign(again=1)
f = first.merge(second, on=['user_id', 'game_id'], how='left').fillna({'again': 0})
print('P(game shown again later | first-exposure rel):'); print(f.groupby('rel1').again.agg(['size', 'mean']).round(3))
# prefix vs slate rel
print('mean rel prefix (train n>=7 pos<=3):', trd[(trd.groupby("session_id").session_id.transform("size") >= 7) & (trd.position <= 3)].rel.mean().round(4), ' slate', Tt.rel.mean().round(4))

# 8. composite sanity (hand-built, no fitting)
print('\n=== HAND-BUILT COMPOSITES, NDCG@5 on val (no fitting; diagnostics only) ===')
for D in (V,):
    z = lambda s: (s - s.mean()) / s.std()
    D['c1'] = z(D['pop'])
    D['c2'] = z(D['pop']) + z(D.aff_fam.fillna(0)) + z(D.aff_mech.fillna(0))
    D['c3'] = D['c2'] + 0.5 * z(D.fr_comp.clip(upper=3)) + 0.5 * z(D.best_prev.fillna(0))
    D['c4'] = D['c3'] + 0.5 * z(D.gwatch.fillna(D.gwatch.mean()))
    D['ps_famgood'] = (D.ps_family_good > 0).astype(int)
    D['c5'] = D['c4'] + 0.3 * z(D.ps_famgood) - 0.3 * z(D.difficulty)
    D['rand'] = np.random.RandomState(0).rand(len(D))
    for c in ['rand', 'c1', 'c2', 'c3', 'c4', 'c5']:
        print(c, [round(x, 4) if isinstance(x, float) else x for x in ndcg(D, c)])
# 'perfect knowledge' of user's prefix? score by u_rate only (constant within session -> random)
print('NDCG@5 of user-level only (should ~ random):', ndcg(V.assign(u=V.u_rate.fillna(0) + V.rand * 1e-6), 'u'))
