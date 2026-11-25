from load import *
pd.set_option('display.width',200); pd.set_option('display.max_columns',50)
t=tr.drop_duplicates().merge(games,on='game_id')
t['n']=t.groupby('session_id').session_id.transform('size')
def summ(g): return pd.Series({'n':len(g),'mean_rel':g.rel.mean(),'r>=1':(g.rel>=1).mean(),'r>=2':(g.rel>=2).mean(),'r3':(g.rel==3).mean(),'play_now':g.play_now.mean()})
print('REL BY POSITION (train, all)'); 
t['posb']=pd.cut(t.position,[0,1,2,3,4,5,6,8,10,15,20,50]); print(t.groupby('posb',observed=True).apply(summ).round(4))
print('REL BY POSITION restricted sessions n>=7'); print(t[t.n>=7].groupby('posb',observed=True).apply(summ).round(4))
# within session relative position
t['relpos']=t.position/t.n
print(t[t.n>=7].groupby(pd.cut(t.relpos,[0,.2,.4,.6,.8,1.0]),observed=True).apply(summ).round(4))
# session length effect
print('BY SESSION LENGTH'); print(t.groupby(pd.cut(t.n,[0,5,6,9,12,16,20,50]),observed=True).apply(summ).round(4))
# play_now at pos 1..3 vs session continuing?
# session length vs engagement in prefix
# fatigue: cumulative plays so far in session
t=t.sort_values(['session_id','position'])
t['cum_plays']=t.groupby('session_id').play_now.cumsum()-t.play_now
print('BY previous plays in session'); print(t[t.n>=7].groupby(t.cum_plays.clip(upper=6)).apply(summ).round(4))
t['cum_good']=t.groupby('session_id').rel.transform(lambda s:(s>=1).cumsum()-(s>=1))
print('BY previous good in session'); print(t[t.n>=7].groupby(t.cum_good.clip(upper=5)).apply(summ).round(4))
# popularity vs exposure: game train rate vs exposure share
gp=t.groupby('game_id').agg(imps=('rel','size'),rate=('rel',lambda s:(s>=1).mean()),added=('added_day','first'))
print('corr(imps, rate) all',gp.imps.corr(gp.rate,method='spearman'))
old=gp[gp.added==0]; print('corr (added 0 only)',old.imps.corr(old.rate,method='spearman'), len(old))
gp['q']=pd.qcut(gp.rate,5,duplicates='drop'); print(gp.groupby('q',observed=True).imps.agg(['size','mean','median']))
# exposure by first-half vs second-half of train: popularity at day<12 vs exposure after
h1=t[t.day<12].groupby('game_id').rel.apply(lambda s:(s>=1).mean()); e2=t[t.day>=12].game_id.value_counts()
x=pd.concat([h1.rename('rate1'),e2.rename('exp2')],axis=1).dropna(); print('corr past rate -> later exposure',x.rate1.corr(x.exp2,method='spearman'))
# exposure by editorial_stars, difficulty, family
for c in ['editorial_stars','difficulty','family','mechanic','has_leaderboard','pace','art_style','theme']:
    ex=t.groupby(c,dropna=False).size()/ t.shape[0]; ng=games.groupby(c,dropna=False).size()/200
    r=t.groupby(c,dropna=False).apply(summ)[['mean_rel','r>=1','r>=2']]
    print('\n',c); print(pd.concat([(ex/ng).rename('exposure_ratio'),r.round(4)],axis=1))
# new games exposure over time
newg=games[games.added_day>0]
print('NEW GAMES: impressions per day since add')
tn=t[t.added_day>0].copy(); tn['age']=tn.day-tn.added_day
print(tn.groupby(pd.cut(tn.age,[-1,0,1,2,4,7,10,25]),observed=True).apply(summ).round(4))
per=tn.groupby('game_id').agg(added=('added_day','first'),imps=('rel','size'),rate=('rel',lambda s:(s>=1).mean()))
per['days_avail']=24-per.added; per['imps_per_day']=per.imps/per.days_avail; print(per.sort_values('added'))
# per-day impression per old game average
old_pd=t[t.added_day==0].groupby('day').size()/170; new_pd=t[t.added_day>0].groupby('day').size()/ (games.added_day.between(1,23).astype(int).sum()) 
print('old imps per game per day',old_pd.round(1).to_dict())
av=[(d,(games.added_day.between(1,d)).sum()) for d in range(24)]
nd=t[t.added_day>0].groupby('day').size(); print('new games per-game-per-day',{d:round(nd.get(d,0)/max(1,c),1) for d,c in av})
# position of new games within sessions
print('mean position new vs old', t[t.added_day>0].position.mean(), t[t.added_day==0].position.mean())
print('new games share of pos<=3', (t[t.position<=3].added_day>0).mean(), 'pos>3',(t[t.position>3].added_day>0).mean())
# per game: mean position
gm=t.groupby('game_id').position.mean(); print('spearman(game mean pos, rate)',gm.corr(gp.rate,method='spearman'))
