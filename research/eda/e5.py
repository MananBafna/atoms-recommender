from load import *
import sys
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 200)
Tt = pd.read_pickle('train_feats.pkl')
V = pd.read_pickle('val_feats.pkl')
trd = tr.drop_duplicates().copy()
G = games.set_index('game_id')

for D in (Tt, V):
    D['sess_mean'] = D.groupby('session_id').rel.transform('mean')
    D['resid'] = D.rel - D.sess_mean
    D['hour'] = (D.ts % 86400) / 3600.0
    D['platform'] = D.user_id.map(users.set_index('user_id').platform)
    D['age'] = D.day - D.added_day
    D['g'] = (D.rel >= 1).astype(int)

# game popularity: LOO in train, train-only for val
gs = trd.groupby('game_id').agg(n=('rel', 'size'), good=('rel', lambda s: (s >= 1).sum()), comp=('rel', lambda s: (s >= 2).sum()), rsum=('rel', 'sum'))
glob = (trd.rel >= 1).mean()
def pop_rate(D, loo):
    n = D.game_id.map(gs.n).fillna(0)
    gd = D.game_id.map(gs.good).fillna(0)
    if loo:
        n = n - 1
        gd = gd - D.g
    return (gd + 10 * glob) / (n + 10)
Tt['pop'] = pop_rate(Tt, True)
V['pop'] = pop_rate(V, False)
# global mechanic/family rates from train
for c in ['mechanic', 'family', 'theme', 'art_style', 'pace']:
    gm = trd.assign(c=trd.game_id.map(G[c])).groupby('c').rel.apply(lambda s: (s >= 1).mean())
    for D in (Tt, V):
        D['g_' + c] = D[c].map(gm)
K = 5
for D in (Tt, V):
    D['aff_mech'] = (D.mech_good + K * D.g_mechanic) / (D.mech_n + K) - D.g_mechanic
    D['aff_fam'] = (D.fam_good + K * D.g_family) / (D.fam_n + K) - D.g_family
    D['aff_theme'] = (D.theme_good + K * D.g_theme) / (D.theme_n + K) - D.g_theme
    D['aff_art'] = (D.art_good + K * D.g_art_style) / (D.art_n + K) - D.g_art_style
    D['aff_pace'] = (D.pace_good + K * D.g_pace) / (D.pace_n + K) - D.g_pace
    D['mech_raw'] = D.mech_good / D.mech_n.replace(0, np.nan)

def tab(D, col, bins=None, label=None, extra=False, minn=0):
    x = D.copy()
    key = pd.cut(x[col], bins) if bins is not None else x[col]
    g = x.groupby(key, observed=True, dropna=False)
    r = pd.DataFrame({'n': g.size(), 'mean_rel': g.rel.mean(), 'P(r>=1)': g.rel.apply(lambda s: (s >= 1).mean()),
                      'P(r>=2)': g.rel.apply(lambda s: (s >= 2).mean()), 'P(r=3)': g.rel.apply(lambda s: (s == 3).mean()),
                      'resid': g.resid.mean()})
    if extra and 'play_now' in x:
        r['play_now'] = g.play_now.mean()
        r['quit|play'] = g.apply(lambda d: ((d.play_now == 1) & (d.play_seconds < 20)).sum() / max(1, (d.play_now == 1).sum()))
    return r.round(4)

def both(col, bins=None, extra=True):
    print(f'\n--- {col} --- TRAIN')
    print(tab(Tt, col, bins, extra=extra))
    print(f'--- {col} --- VAL')
    print(tab(V, col, bins))

def pair_auc(D, col, sign=1):
    """within-session pairwise concordance: among pairs with different rel, P(higher-rel item has higher signal); ties 0.5."""
    num = 0.0
    den = 0.0
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

print('=== SIGNAL TABLES ===')
both('difficulty')
both('typical_session_sec', [0, 60, 80, 100, 120, 200])
both('editorial_stars')
V['stars_f'] = V.editorial_stars.fillna(-1)
Tt['stars_f'] = Tt.editorial_stars.fillna(-1)
both('has_leaderboard')
for c in ['mechanic', 'family', 'theme', 'art_style', 'pace']:
    both(c, extra=False)
both('age', [-1, 0, 2, 5, 10, 30])
both('platform')
both('hour', [0, 4, 8, 12, 16, 20, 24])
both('day')
print('\n=== pop (leave-one-out train rate) deciles ===')
both('pop', pd.qcut(Tt['pop'], 5, retbins=True)[1])
# SESSION structure
print('\n=== sessions: share of eval sessions with no relevant slate item (val) ===')
sm = V.groupby('session_id').rel.max()
print('val sessions',len(sm),'no relevant',(sm==0).sum(),'max rel dist',sm.value_counts().sort_index().to_dict())
print('slate mean rel by session; variance share between sessions: ',
      round(V.sess_mean.var() / V.rel.var(), 3), '(train', round(Tt.sess_mean.var() / Tt.rel.var(), 3), ')')

# variance decomposition by user
print('user-level ICC approx (train eval-like): var of user mean / total', round(Tt.groupby('user_id').rel.transform('mean').var() / Tt.rel.var(), 3))

# user history features
print('\n=== USER HISTORY FEATURES ===')
for D in (Tt, V):
    D['u_skill_b'] = pd.cut(D.u_skill, [0, 55, 60, 65, 70, 100])
both('u_n', [-1, 0, 10, 30, 80, 1e9])
both('u_sessions', [-1, 0, 2, 5, 10, 40])
both('u_rate', [-1, 0.05, 0.10, 0.15, 0.2, 1])
print('\n--- affinity (mechanic, vs global mechanic rate), personalised deviation ---')
for col in ['aff_mech', 'aff_fam', 'aff_theme', 'aff_art', 'aff_pace']:
    both(col, [-1, -0.05, -0.02, 0.02, 0.05, 1], extra=False)
print('\n--- mech_n (user history volume in this mechanic) x high-affinity; val ---')
V['mechn_b'] = pd.cut(V.mech_n, [-1, 0, 3, 10, 30, 1e9])
Tt['mechn_b'] = pd.cut(Tt.mech_n, [-1, 0, 3, 10, 30, 1e9])
both('mechn_b', extra=False)
# raw mechanic rate
print('\n--- user raw past rate in mechanic (n>=5) ---')
for D in (Tt, V):
    D['mraw'] = np.where(D.mech_n >= 5, D.mech_raw, np.nan)
both('mraw', [-0.01, 0.02, 0.08, 0.15, 0.25, 1.0], extra=False)

# SKILL x difficulty
print('\n=== SKILL x DIFFICULTY ===')
def skill_cross(D, name):
    d = D.dropna(subset=['u_skill']).copy()
    d = d[d.u_skill_n >= 2]
    d['skill_t'] = pd.qcut(d.u_skill, 3, labels=['low', 'mid', 'high'])
    d['diff_b'] = pd.cut(d.difficulty, [0, 2, 3, 5], labels=['1-2', '3', '4-5'])
    g = d.groupby(['skill_t', 'diff_b'], observed=True)
    r = pd.DataFrame({'n': g.size(), 'mean_rel': g.rel.mean(), 'P(r>=1)': g.g.mean(), 'P(r>=2)': g.rel.apply(lambda s: (s >= 2).mean())})
    if 'play_now' in d:
        r['play_now'] = g.play_now.mean()
        r['quit|play'] = g.apply(lambda x: ((x.play_now == 1) & (x.play_seconds < 20)).sum() / max(1, (x.play_now == 1).sum()))
        r['comp|play'] = g.apply(lambda x: (x.completed == 1).sum() / max(1, (x.play_now == 1).sum()))
    print(name, 'n with skill', len(d), 'terciles of u_skill:', d.u_skill.quantile([1/3, 2/3]).round(1).tolist())
    print(r.round(4))
skill_cross(Tt, 'TRAIN')
skill_cross(V, 'VAL')
# also user's skill vs diff on has_leaderboard=0 games
print('\n=== skill source: users with any score in history, by leaderboard ===')
print('train eval-like rows with u_skill known', Tt.u_skill.notna().mean().round(3), ' val', V.u_skill.notna().mean().round(3))
print('val has_leaderboard=1 share', V.has_leaderboard.mean())

# FRIENDS
print('\n=== FRIENDS (friend played >=1 / completed (>=2) this game earlier) ===')
for D in (Tt, V):
    D['fr_good_b'] = D.fr_good.clip(upper=3)
    D['fr_comp_b'] = D.fr_comp.clip(upper=3)
both('fr_good_b', extra=False)
both('fr_comp_b', extra=False)
# friend effect after controlling pop: residual of rel vs pop
for nm, D in (('train', Tt), ('val', V)):
    x = D.assign(any_fr=(D.fr_comp > 0))
    print(nm, 'friend>=2 ; mean pop', x.groupby('any_fr')['pop'].mean().round(4).to_dict(), ' P(r>=1)', x.groupby('any_fr').g.mean().round(4).to_dict(), 'n', x.groupby('any_fr').size().to_dict())

# REPEAT EXPOSURE
print('\n=== REPEAT EXPOSURE (user saw this game before) ===')
for D in (Tt, V):
    D['seen_b'] = D.seen_n.clip(upper=4)
both('seen_b')
for D in (Tt, V):
    D['best_prev_f'] = D.best_prev.fillna(-1)
both('best_prev_f', extra=False)
both('seen_days_ago', [-1, 1, 3, 7, 30])

# WITHIN SESSION
print('\n=== WITHIN SESSION: prefix outcomes ===')
both('p_good', extra=False)
both('p_quit', extra=False)
for D in (Tt, V):
    D['ps_fam_state'] = np.where(D.ps_family_n == 0, 'none', np.where(D.ps_family_good > 0, 'good', np.where(D.ps_family_quit > 0, 'quit', 'seen-noplay')))
    D['ps_mech_state'] = np.where(D.ps_mechanic_n == 0, 'none', np.where(D.ps_mechanic_good > 0, 'good', np.where(D.ps_mechanic_quit > 0, 'quit', 'seen-noplay')))
    D['ps_fam_comp'] = np.where(D.ps_family_n == 0, 'none', np.where(D.ps_family_comp > 0, 'completed', 'not-completed'))
both('ps_fam_state', extra=False)
both('ps_mech_state', extra=False)
both('ps_fam_comp', extra=False)
# does same-family prefix good matter relative to p_good baseline: residual column already in tab
# pairwise AUC table
print('\n=== WITHIN-SESSION PAIRWISE CONCORDANCE (0.5 = no signal) ===')
sig = [('pop', 1), ('editorial_stars', 1), ('difficulty', -1), ('typical_session_sec', -1), ('has_leaderboard', 1), ('age', 1),
       ('aff_mech', 1), ('aff_fam', 1), ('aff_theme', 1), ('aff_art', 1), ('aff_pace', 1), ('fr_good', 1), ('fr_comp', 1), ('seen_n', 1),
       ('best_prev', 1), ('g_mechanic', 1), ('g_family', 1), ('ps_family_good', 1), ('ps_family_quit', -1), ('ps_mechanic_good', 1), ('ps_mechanic_quit', -1)]
rows = []
for c, s in sig:
    rows.append((c, round(pair_auc(Tt, c, 1), 4), round(pair_auc(V, c, 1), 4), round(pair_auc(Tt, c, -1), 4)))
print(pd.DataFrame(rows, columns=['signal', 'train_auc(+)', 'val_auc(+)', 'train_auc(-)']).to_string())
Tt.to_pickle('train_feats2.pkl')
V.to_pickle('val_feats2.pkl')
