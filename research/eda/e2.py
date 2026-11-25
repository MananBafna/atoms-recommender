from load import *
pd.set_option('display.width',200); pd.set_option('display.max_columns',50)
vsess=set(vs.session_id); psess=set(vp.session_id); esess=set(ve.session_id); lsess=set(vl.session_id)
print('val sessions: slates',len(vsess),'prefix',len(psess),'events',len(esess),'prefix==slates',psess==vsess)
print('slate sess in events',len(vsess&esess),'events sess not in slates',len(esess-vsess))
# events rows for slate sessions
ev_sl=ve[ve.session_id.isin(vsess)]
print('val_events rows in evaluated sessions',len(ev_sl),'of',len(ve))
m=ve.merge(vs[['session_id','game_id']].assign(inslate=1),on=['session_id','game_id'],how='left')
print('val_events rows matching slate (sid,gid)',m.inslate.sum())
m2=ve.merge(vp[['session_id','position','game_id']].assign(inpre=1),on=['session_id','position','game_id'],how='left')
print('val_events rows matching prefix rows',m2.inpre.sum(), 'prefix rows',len(vp))
# are prefix rows identical to events rows?
pe=vp.merge(ve,on=['session_id','position'],suffixes=('','_e'))
print('prefix rows found in events by (sid,pos)',len(pe),' game same',(pe.game_id==pe.game_id_e).mean(),' watch same',((pe.watch_ms==pe.watch_ms_e)|(pe.watch_ms.isna()&pe.watch_ms_e.isna())).mean())
# slate games per evaluated session in events by pos>3?
print('events positions>3 in eval sessions',(ev_sl.position>3).sum(), 'positions<=3',(ev_sl.position<=3).sum())
print('events sessions total',ve.session_id.nunique(),' with >=7 imps',(ve.groupby('session_id').size()>=7).sum())
# labels derivable?
lab=vl.merge(ve,on=['session_id','game_id'],how='left'); print('label rows with event row',lab.user_id.notna().sum())
print('val label session count by size of events session')
sz=ve.groupby('session_id').size()
print('session length for eval sessions in events: min',sz[list(vsess&esess)].min() if vsess&esess else None)
print('session impressions: val_events length vs prefix+slate'); 
cnt=vs.groupby('session_id').size()+3
print('slate+3 stats',cnt.describe())
# val_events sessions not in slates: how many and why
ne=ve[~ve.session_id.isin(vsess)].groupby('session_id').size(); print('non-eval val_events sessions',len(ne),'len dist',ne.describe())
# test
tsess=set(ts_.session_id); print('test sessions',len(tsess),'in val_events',len(tsess&esess),'in train',len(tsess&set(tr.session_id)),'val-eval in train',len(vsess&set(tr.session_id)))
print('test sess ids range',min(tsess),max(tsess),' val',min(vsess),max(vsess),' train max',tr.session_id.max(),' ve',ve.session_id.min(),ve.session_id.max())
print('val_events days',ve.day.min(),ve.day.max(),'test prefix day',tp.day.min(),tp.day.max(),'val prefix day',vp.day.min(),vp.day.max())
print('val slates ts vs events', (vs.ts==vs.session_id.map(vp.groupby('session_id').ts.first())).mean())
# is test prefix in val_events? any rows
print('test prefix (sid) in val_events',tp.session_id.isin(esess).sum())
# user overlap
print('users in val slates',vs.user_id.nunique(),'test',ts_.user_id.nunique(),'train',tr.user_id.nunique(),'val_events',ve.user_id.nunique())
# eval session sessions: exact timestamps—are val events after-session? 
# evaluated sessions' labels vs val_events outcomes
ev=ve.copy()
# check slate game relevance from events
x=vl.merge(ev[['session_id','game_id','rel']],on=['session_id','game_id'],how='left',suffixes=('','_ev'))
print('labels with event match',x.rel_ev.notna().sum(), 'agree',(x.relevance==x.rel_ev).sum())
# eval sessions overall: val_events sessions that are eval
print('# eval sessions among val_events sessions', len(vsess&esess), ' sessions in val_events w/ >=7',(sz>=7).sum())
# sessions per day distribution overall
# min length for slate sessions
print('prefix rows per session',vp.groupby('session_id').size().value_counts().to_dict())
# later sessions of same user in val_events for val_eval user (temporal leakage for val): events from sessions after eval session
