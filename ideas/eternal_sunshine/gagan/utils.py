import numpy as np
from PIL import Image
import torch
from torchvision import models
import torchvision.transforms as T
import os

def get_image(path, crop = []):
    img = Image.open(path)
    img = img.rotate(90, expand = 1)
    return img.crop(crop)


def get_img_texture_features(path2img, crop = (50, 300, 2700, 4200), resize_to = [1200,800], box_size = [200,300], npatches = 20, patch_size = 30, verbose = True):
    img = get_image(path2img, crop)
    # resize ?
    img = img.resize(resize_to, resample = 1)
    img_list = crop_grid(img, box_size = box_size, top_offset = 100)
    if verbose:
        print('image block size : %d, %d'%(img_list[0].size))
    img_patches = [generate_random_img_patches(I, patch_size=patch_size, num_locs=npatches) for I in img_list]
    gclm_feats = np.stack([get_GCLM_features(p) for p in img_patches]).flatten()
    return gclm_feats


def get_all_texture_mapping_features(path_to_images, crop = (50, 300, 2700, 4200) , resize_to = [800,1200], 
                                     box_size=[600,800], npatches = 20, patch_size = 30, verbose_every = 20):
    dirs = os.listdir(path_to_images)
    
    X = []
    cnt = 0
    for d in dirs:
        fls = os.listdir(os.path.join(path_to_images, d))
        for f in fls:
            path2img = os.path.join(path_to_images, d, f)
            # check file size
            b = float(os.path.getsize(path2img))
            if b < 300e3:
                # blank dark image
                continue
            x = get_img_texture_features(path2img, crop, resize_to, box_size, npatches, patch_size, verbose=False)
            cnt += 1
            X.append(x)
            if cnt % verbose_every ==0:
                print(' done with %d images , feature space size %d '%(cnt, x.shape[-1]))
    return X


def crop_grid(img, box_size = [900,500], top_offset = 0):
    # can you split the image into small boxes of this size ? 
    H,W = img.size
    nrows = int(np.floor(H / box_size[0]))
    ncols = int(np.floor(W / box_size[1]))
    imgs = []
    
    left = 0
    up = 0
    low = up + box_size[1]
    right = left + box_size[0]
    for i in range(nrows):
        for j in range(ncols):
            I = img.crop((left, up, right, low))
            imgs.append(I)
            left += box_size[0]
            right = left + box_size[0]
        up += box_size[1]
        low = up + box_size[1]
        left = 0
        right = left + box_size[0]
    return imgs


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def generate_random_img_locations(img, patch_size = 5, num_locs = 5):
    # over sample row and column start indices
    H,W = img.size
    idx_rows = np.random.choice(np.arange(patch_size, H-patch_size), size = num_locs, replace=False)
    idx_cols = np.random.choice(np.arange(patch_size, W-patch_size), size = num_locs, replace=False)
    locations = [(r,c) for (r,c) in zip(idx_rows, idx_cols)]
    patches = []
    for loc in locations:
        patches.append(np.array(img.crop([loc[0], loc[1], loc[0]+patch_size, loc[1]+patch_size] )))
    return patches


def generate_random_img_patches(img, patch_size = 5, num_locs = 5):
    # over sample row and column start indices and return numpy array
    im = np.array(img)
    H = im.shape[0]
    W = im.shape[1]
    idx_rows = np.random.choice(np.arange(patch_size, H-patch_size), size = num_locs, replace=True)
    idx_cols = np.random.choice(np.arange(patch_size, W-patch_size), size = num_locs, replace=True)
    locations = [(r,c) for (r,c) in zip(idx_rows, idx_cols)]
    patches = []
    for loc in locations:
        I = im[loc[0] : loc[0]+patch_size, loc[1]:loc[1]+patch_size]
        if I.shape[1] < patch_size:
            pdb.set_trace()
        patches.append(I)
    return patches


def get_GCLM_features(patches):
    # compute some GLCM properties each patch
    xs = np.zeros((len(patches),1))
    ys = np.zeros((len(patches),1))
    for i,patch in enumerate(patches):
        # convert patch to grayscale
        patch = rgb2gray(patch).astype('uint8')
        glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
        xs[i] = greycoprops(glcm, 'dissimilarity')[0, 0]
        ys[i] = greycoprops(glcm, 'correlation')[0, 0]
    return np.concatenate([xs, ys],axis=1).flatten()