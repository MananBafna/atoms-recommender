from load import *
pd.set_option('display.width',200); pd.set_option('display.max_columns',50)
print(tr.describe().T)
for n,d in [('train',tr),('val_events',ve)]:
    print('=====',n,len(d))
    print('exact dup',d.duplicated().sum(),' (sid,gid) dup',d.duplicated(['session_id','game_id']).sum(),' (sid,pos) dup',d.duplicated(['session_id','position']).sum())
    print('blank watch',d.watch_ms.isna().mean(),d.watch_ms.isna().sum(),' blank play_sec',d.play_seconds.isna().sum(),d.play_seconds.isna().mean())
    print('blank play_sec by play_now'); print(d.groupby('play_now').play_seconds.apply(lambda s:s.isna().mean()))
    print('blank watch by play_now'); print(d.groupby('play_now').watch_ms.apply(lambda s:s.isna().mean()))
    print('blank ps among completed',d[d.completed==1].play_seconds.isna().mean(), 'n completed',(d.completed==1).sum())
    print('play_now==0 & ps>0',((d.play_now==0)&(d.play_seconds>0)).sum(),' completed&!play_now',((d.completed==1)&(d.play_now==0)).sum(),' replays/share/chal & !completed',(((d.replays>0)|(d.shared>0)|(d.challenged>0))&(d.completed==0)).sum(),' score & !completed',(d.score_percentile.notna()&(d.completed==0)).sum(), 'completed & no score', ((d.completed==1)&d.score_percentile.isna()).sum())
    p=d[d.play_now==1]
    print('play_now rows',len(p),'ps quantiles'); print(p.play_seconds.describe(percentiles=[.01,.05,.1,.25,.5,.75,.9,.95,.99,.999]))
    print('ps<3',(p.play_seconds<3).sum(),'<5',(p.play_seconds<5).sum(),'<10',(p.play_seconds<10).sum(),'<20',(p.play_seconds<20).sum(),' 20+',(p.play_seconds>=20).sum(), 'blank', p.play_seconds.isna().sum())
    print('watch_ms quantiles'); print(d.watch_ms.describe(percentiles=[.01,.5,.9,.99]))
    print('watch>12000',(d.watch_ms>12000).sum(),'neg',(d.watch_ms<0).sum(), 'watch==0',(d.watch_ms==0).sum())
    print('ps>3600',(d.play_seconds>3600).sum(),'max',d.play_seconds.max(), 'neg ps',(d.play_seconds<0).sum())
    print('ps hist for played (bins)'); print(pd.cut(p.play_seconds,[-1,2,5,10,20,60,120,300,600,1800,1e9]).value_counts().sort_index())
    print('completed with ps<20', ((d.completed==1)&(d.play_seconds<20)).sum())
    print('rel dist'); print(d.rel.value_counts(normalize=True).sort_index(), d.rel.value_counts().sort_index())
    print('pos range',d.position.min(),d.position.max(),'day',d.day.min(),d.day.max())
    # blank ps outcome: do they have completed?
    b=d[d.play_seconds.isna()]; print('blank ps rows: play_now',b.play_now.value_counts().to_dict(),'completed',b.completed.value_counts().to_dict(),'replay>0',(b.replays>0).sum())
    # tapped but short with completed
    print('play_now by blank watch'); print(d.groupby(d.watch_ms.isna()).play_now.mean())
    print('watch==0 by play_now'); print(d.groupby('play_now').apply(lambda x:(x.watch_ms==0).mean()))
# user/game keys validity
print('unknown games in train',(~tr.game_id.isin(games.game_id)).sum(),'unknown users',(~tr.user_id.isin(users.user_id)).sum())
print('impressions before signup',(tr.day<tr.user_id.map(users.set_index('user_id').signup_day)).sum())
print('impr before added_day',(tr.day<tr.game_id.map(games.set_index('game_id').added_day)).sum())
print('ts vs day consistent',((tr.ts//86400)==tr.day).mean())
print('dups example'); print(tr[tr.duplicated(keep=False)].sort_values(['session_id','position']).head(6))
print('friends dup',friends.duplicated().sum(),'selfs',(friends.user_id==friends.friend_id).sum())
print(games.describe(include='all').T)
