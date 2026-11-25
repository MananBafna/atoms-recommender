from load import *
x=vl.merge(ve[['session_id','game_id','rel']],on=['session_id','game_id'],how='left',suffixes=('','_ev'))
print('labels with event match',x['rel'].notna().sum(),'agree',(x.relevance==x['rel']).sum(),'of',len(x))
print(pd.crosstab(x.relevance,x['rel']))
pe=vp.merge(ve,on=['session_id','position'],suffixes=('','_e'))
for c in ['watch_ms','play_now','play_seconds','completed','replays','ts']:
    a,b=pe[c],pe[c+'_e']; print(c,'differs',(~((a==b)|(a.isna()&b.isna()))).sum())
d=pe[~((pe.watch_ms==pe.watch_ms_e)|(pe.watch_ms.isna()&pe.watch_ms_e.isna()))]; print(d[['watch_ms','watch_ms_e']].head())
# eval session user history overlap with other val_events from earlier sessions
ev_ts=vp.groupby('session_id').ts.first()
ne=ve[~ve.session_id.isin(set(vs.session_id))]
print('non-eval val sessions: sessions',ne.session_id.nunique(),'rows',len(ne))
# eval session ts relation to non-eval sessions of same user: how many eval sessions have non-eval earlier val sessions
# do val eval sessions' users have later val sessions (future)?
ss=ve.groupby('session_id').agg(user=('user_id','first'),ts=('ts','min'),n=('ts','size')).reset_index()
ss['eval']=ss.n>=7
print(ss.groupby('eval').n.describe())
# train sessions length distribution
st=tr.groupby('session_id').size(); print('train session len',st.describe(), (st>=7).mean(),'frac sessions >=7')
print('train sessions',len(st),'ev sessions',len(ss))
print('overall: short sessions (<7) in train share',(st<7).mean())
print('prefix rel dist (val)',vp.rel.value_counts(normalize=True).sort_index().to_dict())
print('test prefix rel dist',tp.rel.value_counts(normalize=True).sort_index().to_dict())
print('train-eval-like (>=7) sessions in train',(st>=7).sum())
