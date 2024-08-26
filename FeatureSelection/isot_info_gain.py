import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import numpy as np

#reading file
isot=pd.read_csv('data/isot_botnet.csv',index_col=[0])
X=isot[['sTos', 'dTos', 'SrcWin', 'DstWin', 'sHops', 'dHops', 'sTtl', 'dTtl',
       'TcpRtt', 'SynAck', 'AckDat', 'SrcPkts', 'DstPkts', 'SrcBytes',
       'DstBytes', 'SAppBytes', 'DAppBytes', 'Dur', 'TotPkts', 'TotBytes',
       'TotAppByte', 'Rate', 'SrcRate', 'DstRate']]
ig_scores_isot  = mutual_info_classif(X , y)
np.save('info_gain_result/isot_ig_scores.npy',ig_scores_isot)
