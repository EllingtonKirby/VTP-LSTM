from __future__ import print_function, division
from torch.utils.data import DataLoader
from utils import ngsimDataset

trainEpochs = 10
batch_size = 128

trSet = ngsimDataset(grid_file='../../ngsim_us101_D.csv', tracks_file='../../ngsim_us101_T.csv',)
trDataloader = DataLoader(trSet,batch_size=batch_size,shuffle=True,collate_fn=trSet.collate_fn)

for epoch_num in range(trainEpochs):
    print(f'Epoch {epoch_num}')
    for i, data in enumerate(trDataloader):
        hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, vehid, t, ds = data
        print(f'Examining data iteration {i}')
