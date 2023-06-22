# Deeplearning-based feature extraction

import numpy as np
import tensorflow as tf
import collections
import SimpleITK as sitk
from scipy.ndimage import zoom
import os, sys
import pandas as pd
from keras.preprocessing import image

## Set GPU
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Load model
from keras.applications.vgg19 import VGG19, preprocess_input
base_model = VGG19(weights='imagenet', include_top=True)
model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# Load data directory
imgDir = r'L:\basic\divi\jstoker\slicer_pdac\Master Students SS 23\Mattia\keras_feature_extraction'
dirlist = os.listdir(imgDir)

# Read segmentations in NIfTI/NRRD format (ROI)
def loadSegArraywithID(fold,iden):

    path = fold
    pathList = os.listdir(path)
    segPath = [os.path.join(path,i) for i in pathList if ('seg' in i.lower()) & (iden in i.lower())][0]
    seg = sitk.ReadImage(segPath)
    return seg

# Read images
def loadImgArraywithID(fold,iden):

    path = fold
    pathList = os.listdir(path)

    imgPath = [os.path.join(path,i) for i in pathList if ('im' in i.lower()) & (iden in i.lower())][0]
    img = sitk.ReadImage(imgPath)
    return img

# Bounding box (cropping box)
def maskcroppingbox(images_array, use2D=False):
    images_array_2 = np.argwhere(images_array)
    (zstart, ystart, xstart), (zstop, ystop, xstop) = images_array_2.min(axis=0), images_array_2.max(axis=0) + 1
    return (zstart, ystart, xstart), (zstop, ystop, xstop)


extracted_images = []
def featureextraction(imageFilepath,maskFilepath):
    image_array = sitk.GetArrayFromImage(imageFilepath)
    mask_array = sitk.GetArrayFromImage(maskFilepath)
    (zstart, ystart, xstart), (zstop, ystop, xstop) = maskcroppingbox(mask_array, use2D=False)
    roi_images = image_array[zstart-1:zstop+1,ystart:ystop,xstart:xstop].transpose((2,1,0))
    roi_images1 = zoom(roi_images, zoom=[224/roi_images.shape[0], 224/roi_images.shape[1],1], order=3)
    roi_images2 = np.array(roi_images1,dtype=float)
    x = tf.keras.preprocessing.image.img_to_array(roi_images2)
    num = []
    for i in range(zstart,zstop):
        mask_array = np.array(mask_array, dtype='uint8')
        images_array_3 = mask_array[:,:,i]
        num1 = images_array_3.sum()
        num.append(num1)
    maxindex = num.index(max(num))
    x1 = np.asarray(x[:,:,maxindex-1])
    x2 = np.asarray(x[:,:,maxindex])
    x3 = np.asarray(x[:,:,maxindex+1])
    x1 = np.expand_dims(x1, axis=0)
    x2 = np.expand_dims(x2, axis=0)
    x3 = np.expand_dims(x3, axis=0)
    a1 = np.asarray(x1)
    a2 = np.asarray(x2)
    a3 = np.asarray(x3)
    mylist = [a1,a2,a3]
    x = np.asarray(mylist)
    x = np.transpose(x,(1,2,3,0))
    x = preprocess_input(x)
    
     # Save the images before features are extracted
    img_list = [x1.squeeze(), x2.squeeze(), x3.squeeze()]
    
    # Append the img_list to extracted_images list
    extracted_images.append(img_list)

    base_model_pool_features = model.predict(x)

    features = base_model_pool_features[0]

    deeplearningfeatures = collections.OrderedDict()
    for ind_,f_ in enumerate(features):
        deeplearningfeatures[str(ind_)] = f_
    return deeplearningfeatures


import matplotlib.pyplot as plt
def show_image(index):
    """
    Show the image given its index in the extracted_images list.
    
    :param index: int, index of the image in the extracted_images list.
    """
    if index >= len(extracted_images):
        print(f"Index out of range. Total number of images: {len(extracted_images)}")
        return

    img_list = extracted_images[index]
    img_titles = [f"Image {index}, slice {i}" for i in range(len(img_list))]

    fig, axes = plt.subplots(1, len(img_list), figsize=(12, 4))
    for ax, img, title in zip(axes, img_list, img_titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.show()

## Use the show_image function with the desired index
# index = 6  # Change this index to view different images
# show_image(index)

featureDict = {}
for ind in range(len(dirlist)):
    try:
        path = os.path.join(imgDir,dirlist[ind])
        seg = loadSegArraywithID(path,'seg')
        im = loadImgArraywithID(path,'im')

        deeplearningfeatures = featureextraction(im,seg)

        result = deeplearningfeatures
        key = list(result.keys())
        # print(key)
        key = key[0:]

        feature = []
        for jind in range(len(key)):
            feature.append(result[key[jind]])

        featureDict[dirlist[ind]] = feature
        dictkey = key
        print(dirlist[ind])
    except IndexError as ie:
        print(path, "PASSED due to IndexError")
        print(ie)
        pass
    except RuntimeError as re:
        passed.append(path)
        print(path, "PASSED due to RuntimeError")
        print(re)
        pass


# dataframe = pd.DataFrame.from_dict(featureDict, orient='index', columns='dictkey')
dataframe = pd.DataFrame.from_dict(featureDict, orient='index')

#count percentage of non 0 columns
print('average percentage of non-zero features', ((dataframe > 0).sum(axis=1)/dataframe.shape[1]).mean())

#save directory
feature_dir = r"L:\basic\divi\jstoker\slicer_pdac\Master Students SS 23\Mattia\keras_feature_extraction_OUTPUT"
#dataframe.to_csv(feature_dir + "\\vgg_features_eval.csv")