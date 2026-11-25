from load import *
pd.set_option('display.width',200); pd.set_option('display.max_columns',50)
trd=tr.drop_duplicates()
print('train dedup',len(trd))
print('train rel (dedup)',trd.rel.value_counts().sort_index().to_dict(), trd.rel.value_counts(normalize=True).sort_index().round(4).to_dict())
print('val labels',vl.relevance.value_counts().sort_index().to_dict(),vl.relevance.value_counts(normalize=True).sort_index().round(4).to_dict())
print('rel>=1 rate train',(trd.rel>=1).mean(),'val',(vl.relevance>=1).mean())
# train sessions restricted to >=7 and position>3 (like val slates)
tt=trd.merge(trd.groupby('session_id').size().rename('n'),on='session_id'); tte=tt[(tt.n>=7)&(tt.position>3)]
print('train eval-like (n>=7,pos>3) rel',tte.rel.value_counts(normalize=True).sort_index().round(4).to_dict(),len(tte))
# sessions per user
spu=trd.groupby('user_id').session_id.nunique()
print('sessions/user train: users',len(spu),'of',len(users)); print(spu.describe())
print('users with 0 train sessions',(~users.user_id.isin(spu.index)).sum())
print('cold (<=2 sessions) in train hist',(spu<=2).sum(),' + zero-history',(~users.user_id.isin(spu.index)).sum())
print('sessions/user quantiles', spu.quantile([.1,.25,.5,.75,.9]).to_dict())
def sess_hist(extra):
    h=pd.concat([trd]+extra); return h.groupby('user_id').session_id.nunique()
spu_t=sess_hist([ve])
for nm,sl,hist in [('val',vs,spu),('test',ts_,spu_t)]:
    u=sl.groupby('session_id').user_id.first(); n=u.map(spu if nm=='val' else spu_t).fillna(0)
    print(nm,'eval sessions',len(u),'users',u.nunique(),' sessions of cold users (<=2)',(n<=2).sum(),' zero-hist sessions',(n==0).sum(), ' cold users',u[n<=2].nunique())
# impressions per game
ipg=trd.game_id.value_counts().reindex(games.game_id).fillna(0)
print('impressions/game train'); print(ipg.describe()); print('games <30 train',(ipg<30).sum(),'zero',(ipg==0).sum())
hist_t=pd.concat([trd,ve]); ipg_t=hist_t.game_id.value_counts().reindex(games.game_id).fillna(0); print('games<30 train+val',(ipg_t<30).sum(),'zero',(ipg_t==0).sum())
for nm,sl,ip in [('val',vs,ipg),('test',ts_,ipg_t)]:
    c=sl.game_id.map(ip)
    print(nm,'slate rows',len(sl),'games w/ <30 hist',(c<30).sum(),round((c<30).mean(),4),'zero hist',(c==0).sum(),round((c==0).mean(),4),' distinct games <30:',sl.game_id[c<30].nunique(),'zero',sl.game_id[c==0].nunique())
    ad=sl.game_id.map(games.set_index('game_id').added_day); print(nm,' added_day>=25 slate rows',(ad>=25).sum(),'distinct',sl.game_id[ad>=25].nunique(), ' added>=24',(ad>=24).sum())
    ss=sl.groupby('session_id').size(); print(nm,'slate size',ss.describe().round(2).to_dict())
print('games added_day dist'); print(games.added_day.value_counts().sort_index().to_dict())
print('games added>=25',games[games.added_day>=25][['game_id','added_day']].values.tolist())
print('games added 1..24',(games.added_day.between(1,24)).sum())
print('editorial_stars null',games.editorial_stars.isna().sum(), 'among added>=1',games[games.added_day>=1].editorial_stars.isna().sum())
print(games.groupby(games.editorial_stars.isna()).added_day.describe())
# new-game labels in val
vl2=vl.merge(vs[['session_id','game_id']],on=['session_id','game_id']).copy()
vl2['n']=vl2.game_id.map(ipg); vl2['ad']=vl2.game_id.map(games.set_index('game_id').added_day)
print('val rel by hist bucket'); print(vl2.groupby(pd.cut(vl2.n,[-1,0,29,100,300,1e9])).relevance.agg(['size','mean',lambda s:(s>=1).mean(),lambda s:(s>=2).mean()]))
print('val rel by added>=24'); print(vl2.groupby(vl2.ad>=24).relevance.agg(['size','mean',lambda s:(s>=1).mean(),lambda s:(s>=2).mean()]))
print(users.platform.value_counts().to_dict(), 'signup_day>0',(users.signup_day>0).sum(), 'signup>=20',(users.signup_day>=20).sum(),'signup>=24',(users.signup_day>=24).sum())
print('friends per user',pd.concat([friends.user_id,friends.friend_id]).value_counts().describe().round(2).to_dict(),' users with 0 friends',(~users.user_id.isin(pd.concat([friends.user_id,friends.friend_id]))).sum())
print('session len eval-ness: train sessions with n>=7',(trd.groupby('session_id').size()>=7).sum())
# impressions per user per day etc
print('sessions per day train',trd.groupby('day').session_id.nunique().to_dict())
print('val sessions/day',vp.groupby('day').session_id.nunique().to_dict(),'test',tp.groupby('day').session_id.nunique().to_dict())
