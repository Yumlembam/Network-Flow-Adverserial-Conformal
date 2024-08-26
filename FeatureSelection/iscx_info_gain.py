import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import numpy as np

#reading file
iscx=pd.read_csv('../data/ISCX_training.csv',index_col=[0])
X=iscx[['sTos', 'dTos', 'SrcWin', 'DstWin', 'sHops', 'dHops', 'sTtl', 'dTtl',
       'TcpRtt', 'SynAck', 'AckDat', 'SrcPkts', 'DstPkts', 'SrcBytes',
       'DstBytes', 'SAppBytes', 'DAppBytes', 'Dur', 'TotPkts', 'TotBytes',
       'TotAppByte', 'Rate', 'SrcRate', 'DstRate']]
y = iscx['Label']
ig_scores_iscx  = mutual_info_classif(X , y)
np.save('info_gain_result/iscx_ig_scores.npy',ig_scores_iscx)
