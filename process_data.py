import numpy as np
import pandas as pd
import os
import skimage.io
from tqdm.auto import tqdm
import pickle
from model import efficienet_pool, ensemble



sz = 256
N = 16
TRAIN = './kaggle/input/prostate-cancer-grade-assessment/train_images'
MASKS = './kaggle/input/prostate-cancer-grade-assessment/train_label_masks'


def tile(img,mask,image_id):
    result = []
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=255)
    mask = np.pad(mask,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=0)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    mask = mask.reshape(mask.shape[0]//sz,sz,mask.shape[1]//sz,sz,3)
    mask = mask.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        mask = np.pad(mask,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=0)
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    #idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    print(mask.reshape(mask.shape[0],-1).sum(-1).shape)
    idxs = np.argsort(mask.reshape(mask.shape[0],-1).sum(-1))[::-1][:N]
    img = img[idxs]
    mask = mask[idxs]
    for i in range(len(img)):
        result.append({'img':img[i], 'mask':mask[i], 'image_id':image_id})
    return result


if __name__ == '__main__':
    train_df = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')
    names = [name[:-10] for name in os.listdir(MASKS)]
    train_df = train_df[train_df['image_id'].isin(names)]
    # train_df = train_df.groupby('isup_grade').apply(lambda x: x.sample(30)).reset_index(drop=True)
    print(len(train_df))
    results = []
    labels = []
    for i in tqdm(range(len(train_df))):
        img = skimage.io.MultiImage(os.path.join(TRAIN, train_df['image_id'].iloc[i] + '.tiff'))[-1]
        mask = skimage.io.MultiImage(os.path.join(MASKS, train_df['image_id'].iloc[i] + '_mask.tiff'))[-1]
        result = tile(img,mask,train_df['image_id'].iloc[i])
        results.append(result)
        labels.append(train_df['isup_grade'].iloc[i])

    pickle.dump(results, open('results.pkl', 'wb'))
    pickle.dump(labels, open('labels.pkl', 'wb'))

