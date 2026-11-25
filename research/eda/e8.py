from load import *
trd=tr.drop_duplicates()
h=pd.concat([trd,ve]); c=h.game_id.value_counts()
new=games[games.added_day>=25].game_id
print('games added>=25: impressions in train+val_events');print(c.reindex(new).fillna(0).astype(int).to_dict())
print('val_events imps for those:',ve.game_id.value_counts().reindex(new).fillna(0).astype(int).to_dict())
for nm,sl in [('val',vs),('test',ts_)]:
    ad=sl.game_id.map(games.set_index('game_id').added_day)
    print(nm,'rows by added_day>=25 game:',sl[ad>=25].game_id.value_counts().to_dict())
# val prefix/test prefix blanks
for nm,p in [('val_prefix',vp),('test_prefix',tp)]:
    print(nm,'blank ps (tapped)',(p.play_seconds.isna()&(p.play_now==1)).sum(),'blank watch',p.watch_ms.isna().sum(),'of',len(p),' dup',p.duplicated().sum())
# val: users/sessions of eval in signup late
u=users.set_index('user_id')
for nm,sl in [('val',vs),('test',ts_)]:
    s=sl.groupby('session_id').user_id.first(); sd=s.map(u.signup_day)
    print(nm,'eval sessions of users signed up on/after day 20:',(sd>=20).sum(),' >=24',(sd>=24).sum())
# test cold with val_events included
sp=h.groupby('user_id').session_id.nunique()
s=ts_.groupby('session_id').user_id.first(); n=s.map(sp).fillna(0)
print('test sessions cold (<=2 sessions in train+val_events):',(n<=2).sum(),' zero:',(n==0).sum(),'users cold',s[n<=2].nunique())
# val blank resolution: labels for blank play_seconds in val_events
b=ve[ve.play_seconds.isna()].merge(vl,on=['session_id','game_id'],how='inner')
print('val_events blank-ps tapped rows that are slate rows:',len(b),' label dist:',b.relevance.value_counts().sort_index().to_dict(),' completed flag:',b.completed.value_counts().to_dict())
# dup rows: where in file
d=tr[tr.duplicated(keep='first')]; print('dup rows index range',d.index.min(),d.index.max(),' day dist head',d.day.describe()[['min','max']].to_dict())
# dup affect on eval: sessions duplicated: any session in train with duplicates and n>=7
print('dup sessions',d.session_id.nunique(),' positions of dups >3:',(d.position>3).sum())
# does dedup change rel?
print('rel mean with dups',tr.rel.mean().round(4),'dedup',trd.rel.mean().round(4))
# user baseline level variance; users with 0 friends
print('val sessions by # eval sessions per user:',vs.groupby('user_id').session_id.nunique().describe().round(2).to_dict())
