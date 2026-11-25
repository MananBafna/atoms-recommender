from load import *
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 50)
trd = tr.drop_duplicates().copy()
trd['n'] = trd.groupby('session_id').session_id.transform('size')
t = trd.merge(games, on='game_id')
t['quit'] = ((t.play_now == 1) & (t.play_seconds < 20)).astype(int)
print('=== WATCH_MS vs outcome (train, dedup) ===')
t['wb'] = pd.cut(t.watch_ms, [-1, 0, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 11999, 12000])
g = t.groupby('wb', observed=True)
print(pd.DataFrame({'n': g.size(), 'play_now': g.play_now.mean(), 'quit|play': g.apply(lambda d: d.quit.sum() / max(1, d.play_now.sum())),
                    'good|play': g.apply(lambda d: (d.rel >= 1).sum() / max(1, d.play_now.sum())), 'P(r>=1)': g.apply(lambda d: (d.rel >= 1).mean()),
                    'P(r>=2)': g.apply(lambda d: (d.rel >= 2).mean()), 'mean_rel': g.rel.mean()}).round(4))
print('blank watch: play_now', t[t.watch_ms.isna()].play_now.mean().round(4), 'P(r>=1)', (t[t.watch_ms.isna()].rel >= 1).mean().round(4), 'n', t.watch_ms.isna().sum())
print('watch==0: n', (t.watch_ms == 0).sum(), 'play_now', t[t.watch_ms == 0].play_now.mean().round(4))
# does watch relate to session-level/prefix-level engagement or game?
print('corr(watch, play_now) all', t[['watch_ms', 'play_now']].corr().iloc[0, 1].round(4))
print('corr(watch, rel | played)', t[t.play_now == 1][['watch_ms', 'rel']].corr().iloc[0, 1].round(4))
# watch by position
print('mean watch by position bucket'); print(t.groupby(pd.cut(t.position, [0, 1, 2, 3, 5, 8, 12, 50]), observed=True).watch_ms.mean().round(0).to_dict())
# watch vs game attributes: game mean watch vs game play rate
gg = t.groupby('game_id').agg(w=('watch_ms', 'mean'), pn=('play_now', 'mean'), good=('rel', lambda s: (s >= 1).mean()), comp=('rel', lambda s: (s >= 2).mean()))
print('game-level spearman: mean watch vs play_now', gg.w.corr(gg.pn, method='spearman').round(3), ' vs P(r>=1)', gg.w.corr(gg.good, method='spearman').round(3), ' vs P(r>=2)', gg.w.corr(gg.comp, method='spearman').round(3))
# per-user mean watch & user rate
uu = t.groupby('user_id').agg(w=('watch_ms', 'mean'), pn=('play_now', 'mean'), good=('rel', lambda s: (s >= 1).mean()))
print('user-level spearman: mean watch vs play_now', uu.w.corr(uu.pn, method='spearman').round(3), 'vs P(r>=1)', uu.w.corr(uu.good, method='spearman').round(3))
# within-user-game: watch residual
t['w_dm'] = t.watch_ms - t.groupby('user_id').watch_ms.transform('mean')
t['wq'] = pd.qcut(t.w_dm, 5, duplicates='drop')
g = t.groupby('wq', observed=True)
print('watch relative to user mean (quintiles):'); print(pd.DataFrame({'n': g.size(), 'play_now': g.play_now.mean(), 'P(r>=1)': g.apply(lambda d: (d.rel >= 1).mean())}).round(4))
# prefix watch/outcome relation: does mean prefix watch predict slate engagement (session-level)? and does *prefix watch of games predict later play on similar*?
print('\n=== prefix watch vs slate outcome (val) ===')
V = pd.read_pickle('val_feats.pkl') if False else None
vv = vl.merge(vs[['session_id', 'user_id']].drop_duplicates(), on='session_id')
pw = vp.groupby('session_id').agg(pwatch=('watch_ms', 'mean'), pplay=('play_now', 'sum'), pgood=('rel', lambda s: (s >= 1).sum()))
vv = vv.join(pw, on='session_id')
vv['pwb'] = pd.cut(vv.pwatch, [-1, 2000, 3500, 5000, 7000, 12001])
g = vv.groupby('pwb', observed=True)
print(pd.DataFrame({'n': g.size(), 'P(r>=1)': g.relevance.apply(lambda s: (s >= 1).mean()), 'mean_rel': g.relevance.mean()}).round(4))
g = vv.groupby('pgood')
print('by #good in prefix'); print(pd.DataFrame({'n': g.size(), 'P(r>=1)': g.relevance.apply(lambda s: (s >= 1).mean()), 'mean_rel': g.relevance.mean()}).round(4))
# prefix-based session propensity: AUC of session-level
# ---- quits / accidental taps
print('\n=== ACCIDENTAL TAPS / QUICK QUITS ===')
p = t[t.play_now == 1]
print('played rows', len(p), 'quit (<20s)', p.quit.sum(), 'share', p.quit.mean().round(4), ' ps<=10', (p.play_seconds <= 10).sum())
print('watch mean among quits', p[p.quit == 1].watch_ms.mean().round(0), ' among plays>=20', p[p.quit == 0].watch_ms.mean().round(0))
print('quit by watch bucket:'); print(p.groupby(pd.cut(p.watch_ms, [-1, 0, 1500, 3000, 6000, 12000]), observed=True).quit.agg(['size', 'mean']).round(4))
print('quit by difficulty:'); print(p.groupby('difficulty').quit.agg(['size', 'mean']).round(4))
print('quit by typical_session_sec bucket:'); print(p.groupby(pd.cut(p.typical_session_sec, [0, 70, 90, 110, 200]), observed=True).quit.agg(['size', 'mean']).round(4))
print('quit by position:'); print(p.groupby(pd.cut(p.position, [0, 3, 6, 12, 50]), observed=True).quit.agg(['size', 'mean']).round(4))
print('quit by family:'); print(p.groupby('family').quit.agg(['size', 'mean']).round(4))
print('play_now & ps<20 values: ps<=2:', (p.play_seconds < 2.5).sum(), '  2.5-5', ((p.play_seconds >= 2.5) & (p.play_seconds < 5)).sum(), '5-10', ((p.play_seconds >= 5) & (p.play_seconds < 10)).sum(), '10-20', ((p.play_seconds >= 10) & (p.play_seconds < 20)).sum())
print('rel dist of tapped games w/ ps>=20:', p[p.play_seconds >= 20].rel.value_counts().sort_index().to_dict())
# play seconds vs typical_session_sec
pl = p[p.play_seconds >= 20]
print('play_seconds/typical_session_sec mean ratio', (pl.play_seconds / pl.typical_session_sec).describe().round(2).to_dict())
print('completed rate among played>=20 by difficulty'); print(pl.groupby('difficulty').apply(lambda d: pd.Series({'n': len(d), 'completed': (d.completed == 1).mean(), 'rel3|comp': (d[d.completed == 1].rel == 3).mean()})).round(4))
print('has leaderboard completed w/o score:', ((t.completed == 1) & (t.has_leaderboard == 1) & t.score_percentile.isna()).sum(), ' completed & has_lb', ((t.completed == 1) & (t.has_leaderboard == 1)).sum(), ' completed & no lb', ((t.completed == 1) & (t.has_leaderboard == 0)).sum(), 'score present & no lb', (t.score_percentile.notna() & (t.has_leaderboard == 0)).sum())
# replays etc
print('replays>0 while completed=0:', ((t.replays > 0) & (t.completed == 0)).sum(), ' shared>0 & completed 0', ((t.shared > 0) & (t.completed == 0)).sum(), 'challenged&!completed', ((t.challenged > 0) & (t.completed == 0)).sum())
print('rel dist when blank ps (tapped):', t[t.play_seconds.isna()].assign(rel_if_known=lambda d: np.where((d.replays > 0) | (d.shared > 0) | (d.challenged > 0), 3, np.where(d.completed > 0, 2, 1))).rel_if_known.value_counts().sort_index().to_dict(), '(lower bound; 1 means unknown or >=20 or quit)')
# game-level stability
print('\n=== GAME POPULARITY STABILITY ===')
a = t[t.day < 12].groupby('game_id').rel.apply(lambda s: (s >= 1).mean())
b = t[t.day >= 12].groupby('game_id').rel.apply(lambda s: (s >= 1).mean())
x = pd.concat([a, b], axis=1, keys=['a', 'b']).dropna()
print('spearman game P(r>=1) days0-11 vs 12-23:', x.a.corr(x.b, method='spearman').round(3), 'n games', len(x))
vgood = vl.groupby('game_id').relevance.apply(lambda s: (s >= 1).mean()) if 'game_id' in vl else None
vv2 = vl.groupby('game_id').relevance.agg(['size', lambda s: (s >= 1).mean()])
vv2.columns = ['n', 'rate']
tr_rate = t.groupby('game_id').rel.apply(lambda s: (s >= 1).mean())
y = vv2.join(tr_rate.rename('tr'), how='inner')
y = y[y.n >= 20]
print('spearman game P(r>=1): train vs val labels (games n>=20 in val):', y.rate.corr(y.tr, method='spearman').round(3), len(y))
# game rate vs editorial stars etc. correlation among old games
gg2 = games.set_index('game_id').join(tr_rate.rename('rate')).join(t.groupby('game_id').rel.mean().rename('mrel'))
print('spearman game mean_rel vs: ', {c: round(gg2[c].corr(gg2.mrel, method='spearman'), 3) for c in ['difficulty', 'typical_session_sec', 'has_leaderboard', 'editorial_stars', 'added_day']})
# time of day
print('\n=== TIME OF DAY / PLATFORM (train, session-level) ===')
t['hour'] = (t.ts % 86400) // 3600
t['platform'] = t.user_id.map(users.set_index('user_id').platform)
g = t.groupby(pd.cut(t.hour, [-1, 5, 11, 17, 21, 24]), observed=True)
print(pd.DataFrame({'n': g.size(), 'play_now': g.play_now.mean(), 'P(r>=1)': g.apply(lambda d: (d.rel >= 1).mean()), 'mean_rel': g.rel.mean()}).round(4))
g = t.groupby('platform')
print(pd.DataFrame({'n': g.size(), 'play_now': g.play_now.mean(), 'P(r>=1)': g.apply(lambda d: (d.rel >= 1).mean()), 'mean_rel': g.rel.mean()}).round(4))
t['dow'] = t.day % 7
g = t.groupby('dow')
print(pd.DataFrame({'n': g.size(), 'P(r>=1)': g.apply(lambda d: (d.rel >= 1).mean()), 'mean_rel': g.rel.mean()}).round(4))
# hour x family
t['hb'] = pd.cut(t.hour, [-1, 5, 11, 17, 21, 24], labels=['night', 'morning', 'afternoon', 'evening', 'late'])
print((t.groupby(['hb', 'family'], observed=True).rel.apply(lambda s: (s >= 1).mean()).unstack()).round(3))
t['dbin'] = pd.cut(t.difficulty, [0, 2, 3, 5], labels=['1-2', '3', '4-5'])
print((t.groupby(['hb', 'dbin'], observed=True).rel.apply(lambda s: (s >= 1).mean()).unstack()).round(3))
print((t.groupby(['platform', 'family'], observed=True).rel.apply(lambda s: (s >= 1).mean()).unstack()).round(3))
print((t.groupby(['platform', 'dbin'], observed=True).rel.apply(lambda s: (s >= 1).mean()).unstack()).round(3))
# user session hour consistency
