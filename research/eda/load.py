import pandas as pd, numpy as np
import os
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','data')+os.sep
def rel(df):
    pn=df.play_now.fillna(0); ps=df.play_seconds.fillna(0)
    r=np.where((pn<1)|(ps<20),0,np.where((df.replays.fillna(0)>0)|(df.shared.fillna(0)>0)|(df.challenged.fillna(0)>0),3,np.where(df.completed.fillna(0)>0,2,1)))
    return r
rd=lambda n: pd.read_csv(D+n+'.csv')
games=rd('games'); users=rd('users'); friends=rd('friends')
tr=rd('train_impressions'); ve=rd('val_events'); vp=rd('val_prefix'); vs=rd('val_slates'); vl=rd('val_labels'); tp=rd('test_prefix'); ts_=rd('test_slates')
for d in (tr,ve,vp,tp): d['rel']=rel(d)
