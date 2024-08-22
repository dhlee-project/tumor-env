
os.chdir('./data/tissue/')
tissue_dir = glob.glob('NCT-CRC-HE-100K/*/*')
data = [[i,os.path.basename(i).split('-')[0],tissue_class[os.path.basename(i).split('-')[0]] ] for i in tissue_dir]
import pandas as pd
import numpy as np
np.random.seed(777)
data = pd.DataFrame(data, columns=['path','class','label'])
data['type'] = 'Train'

for i in range(8):
    sub_data = data[data['label'] == i]
    idx = sub_data.index.values

    np.random.shuffle(idx)
    thres = int(len(idx)*0.9)
    tr_idx = idx[:thres]
    tt_idx = idx[thres:]
    data.iloc[tt_idx, 3] = 'Test'

data.to_csv('data-norm.csv',index=False)