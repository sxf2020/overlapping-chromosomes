# Individual chromosome images were down-sampled by a factor of 2, rotated 18 times by 2π / 18,
# translated 9 times horizontally and vertically and overlapped to produce samples.
# There are 90667 overlapping chromosome samples in this dataset.

import h5py
import sys
import itertools
import numpy as np
from scipy import ndimage as nd
import skimage as sk
from skimage import segmentation as skg
from skimage.measure import label
from skimage import io
import cv2
from matplotlib import pyplot as plt

from subprocess import check_output
cy3_12 = nd.imread('images/inputImages/Cy3.tif')[50:900,300:1150]
dapi_12 = nd.imread('images/inputImages/DAPI.tif')[50:900,300:1150]

plt.figure(figsize=(13, 15))
plt.subplot(121, xticks=[], yticks=[])
plt.imshow(dapi_12,  interpolation = 'nearest', cmap=plt.cm.gray)
plt.subplot(122,xticks=[], yticks=[])
plt.imshow(cy3_12, interpolation = 'nearest', cmap=plt.cm.gray)

#Extracting patch of an image knowing a label.
def extractParticles_2(greyIm, LabIm):
    LabelImg= LabIm
    GreyImg = greyIm
    locations = nd.find_objects(LabelImg)

    i=1
    extracted_images=[]
    for loc in locations:
        
        lab_image = np.copy(LabelImg[loc])
        grey_image = np.copy(GreyImg[loc])
        
        lab_image[lab_image != i] = 0
        grey_image[lab_image != i] = 0
        extracted_images.append(grey_image)
        i=i+1
    return extracted_images

#Resizing images.
def ResizeImages(ImList):
        '''Find the largest width and height of images belonging to a list.
        Return a list of images of same width/height
        '''
        maxwidth=0
        maxheight=0
        if len(np.shape(ImList[0]))==3:
            components = np.shape(ImList[0])[2]
        imtype = ImList[0].dtype
        for i in range(len(ImList)):
            width=np.shape(ImList[i])[1]
            height=np.shape(ImList[i])[0]
            if width>maxwidth:maxwidth=width
            if height>maxheight:maxheight=height

        NewList=[]
        for i in range(0,len(ImList)):
            width=np.shape(ImList[i])[1]
            height=np.shape(ImList[i])[0]

            diffw=maxwidth-width
            startw=round(diffw/2)
            diffh=maxheight-height
            starth=int(round(diffh/2))
            startw=int(round(diffw/2))
            if len(np.shape(ImList[0]))==3:
                newIm=np.zeros((maxheight,maxwidth,components), dtype=imtype)
                newIm[starth:starth+height,startw:startw+width,:]=ImList[i][:,:,:]
                NewList.append(newIm)
            if len(np.shape(ImList[0]))==2:
                newIm=np.zeros((maxheight,maxwidth), dtype=imtype)
                # center in the big new image
                newIm[starth:starth+height,startw:startw+width]=ImList[i][:,:]
                NewList.append(newIm)
        return NewList

#Generating overlapping chromosomes from a pair of single chromosomes.
def clip_img_to_bounding_box(img):
    bb = nd.find_objects(img[:,:,-1]>0)
    slice0 = bb[0][0]
    slice1= bb[0][1]
    clip = img[slice0,slice1]
    return clip

def patch_to_square(image, seepatch=False):
    if seepatch==True:
        s1=2
        s2=3
    else:
        s1=0
        s2=0
    row = image.shape[0]
    col = image.shape[1]
    Hyp = int(np.ceil(np.sqrt(row**2+col**2)))+1
    drow = int(np.ceil(Hyp-row)/2)
    dcol = int(np.ceil(Hyp-col)/2),
    patch_h= s1*np.ones((row,dcol),dtype=int)
    patch_v= s2*np.ones((drow,col+2*(dcol)), dtype=int)
    e2 = np.hstack((patch_h, image, patch_h))
    return np.vstack((patch_v,e2,patch_v))

def rotated_images(image, step_angle, half_turn = False):
    if half_turn == True:
        angle_max = 180
    else:
        angle_max = 360
    angles = np.arange(0, angle_max, step_angle)
    return [clip_img_to_bounding_box(nd.rotate(image, rotation)) for rotation in angles]

def collection_of_pairs_of_rotated_images(image1, image2, step_angle=10, half_turn=False):
    '''Take two images, rotate them all by "step_angle".
    Make all possible pairs by cartesian product
    then resize them such same shape(size)
    '''

    r_images1 = rotated_images(image1, step_angle, half_turn)
    r_images2 = rotated_images(image2, step_angle, half_turn)

    # For example, product(A, B) returns the same as:  ((x,y) for x in A for y in B).
    pairs = itertools.product(r_images1, r_images2)
    r_pairs = [ResizeImages([p[0],p[1]]) for p in pairs]
    return r_pairs

def translate(image, vector):
    image = np.roll(image, int(vector[0]), axis = 0)
    image = np.roll(image, int(vector[1]), axis = 1)
    return image

def add_mask_to_image(image, mask_value = 1):
    image = np.dstack((image, mask_value*(image>0)))
    return image

def merge_and_roll(still_img, move_img, row, col, clip=True):
    '''row, col: assume two numbers in [0,1]
    images:last component supposed to be a mask
    '''
    u=row
    v=col

    target = np.copy(still_img)
    source = np.copy(move_img)

    #bounding boxes
    bb0 = nd.find_objects(target[:,:,-1]>0)
    bb1 = nd.find_objects(source[:,:,-1]>0)
    #won't work if more than two components C0 and C1
    tmp_bb0 = bb0[0][0]
    C_0 = target[bb0[0][0],bb0[0][1]]
    C_1 = source[bb1[0][0],bb1[0][1]]
    #col, row
    c1 = C_1.shape[1]
    r1 = C_1.shape[0]
    c0 = C_0.shape[1]
    r0 = C_0.shape[0]
    comp = C_0.shape[-1]

    still = np.zeros((r1+2*r0,c1+2*c0,comp),dtype=int)
    still[r0+1:r0+r1+1,c0+1:c0+c1+1]= C_1
    move = np.zeros(still.shape, dtype=int)
    move[:r0,:c0]=C_0
    vector = u*(still.shape[0]-r0),v*(still.shape[1]-c0)
    move = translate(move, vector)

    merge = move+still
    if clip==False:
        return merge
    else:
        return clip_img_to_bounding_box(merge)


def overlapping_generator(image1, image2, mask1 = 1, mask2 = 2, 
                          rotation_step = 30, translations_number = 9,turn = False):
    '''
    * This function takes two greyscaled images of chromosomes on black backaground
    * add a mask to each image:
        + same value for each mask (1 and 1) so that the overlapping pixels are set to 2.
        + different values for each mask (1 and 2) so that the overlapping pixels are set to 3.
    * rotate one chromosome of a multiple of a fixed angle in degree and make a pair of chromosomes
    (rotated, fixed position).
    *perform relative translations of one chromosome with a given number
    *returns a list of merged pair of images.
    '''
    c0 = clip_img_to_bounding_box(add_mask_to_image(image1, mask_value = mask1))
    c1 = clip_img_to_bounding_box(add_mask_to_image(image2,mask_value = mask2))

    CR = collection_of_pairs_of_rotated_images(c0, c1, step_angle = rotation_step, half_turn= turn)
    
    #Prepare translation vectors u (same on rows and lines)
    u = np.linspace(0,1,num = translations_number, endpoint = True)[1:-1]
    P = [t for t in itertools.product(u,u)]
    
    overlappings = []
    #Take a pair of images
    for pair_of_images in CR:# CR[::3] Just play with a subset
        im1 = pair_of_images[0]
        im2 = pair_of_images[1]
        #Now translate one image relative to the other one
        for t in P:#P[::3]Just play with a subset
            u = t[0]
            v = t[1]
            overlappings.append(merge_and_roll(im1, im2, u, v))

    overlapping_chrom_learning_set = ResizeImages(overlappings)
    return overlapping_chrom_learning_set

#define structuring elements for further image segmentation
kernel100 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50))
kernel18 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))

#convert the 12 bits images into 8 bits
#easier to handle for segmentation
#needed to make a rgb image
d8 = np.uint8(dapi_12/16.0)
cy8 = np.uint8(cy3_12/16.0)

#Let's make images with background set to zero
dapi_cv = cv2.morphologyEx(d8, cv2.MORPH_TOPHAT, kernel100)
cy3_cv = cv2.morphologyEx(cy8, cv2.MORPH_TOPHAT, kernel18)

dapi_cv = 1.0*dapi_cv - 5
dapi_cv[dapi_cv <0] = 0
dapi_cv = np.uint8(dapi_cv)

cy3_cv = 1.0*cy3_cv - 5
cy3_cv[cy3_cv<0] = 0
cy3_cv = np.uint8(cy3_cv)

#Combine the two images into a rgb image
red = np.uint8(cy3_cv)
green = np.zeros(cy3_cv.shape, dtype=np.uint8)
blue = np.uint8(dapi_cv)

color = np.dstack((red, green, blue))

#Start chromosomes segmentation
seg =  sk.filters.threshold_adaptive(dapi_cv, block_size = 221)
seg = sk.morphology.binary_opening(seg, selem = sk.morphology.disk(5))
sk.segmentation.clear_border(seg, buffer_size=3, in_place= True)
labelled = label(seg)
dapi_cy3 = (1.0*dapi_cv + 1.0*cy3_cv)/2
combined = np.uint8(255*sk.exposure.rescale_intensity(dapi_cy3))
single_chroms = extractParticles_2(combined, labelled)
resized_chroms = ResizeImages(single_chroms)
singles_rgb = extractParticles_2(color, labelled)
resized_rgb = ResizeImages(singles_rgb)

half_karyotype = []

for chrom in resized_chroms:
    half_karyotype.append(chrom[::2, ::2])

print('chromosome num ',len(half_karyotype))
print('Image resolution ',half_karyotype[0].shape)

subset_pairs = []
for pair in itertools.combinations(half_karyotype, 2):
    subset_pairs.append(pair)
print(len(subset_pairs),' pairs of chromosomes generated')
print(type(subset_pairs[0]))
print(subset_pairs[0][0].shape)

candidates_dataset = []
for pair in subset_pairs[::37]:
    im1 = pair[0]
    im2 = pair[1]
    overl = overlapping_generator(im1,im2,rotation_step=20, translations_number=9, turn = True)
    candidates_dataset = candidates_dataset + overl

print(len(candidates_dataset),'pairs of chromesomes')

clean_subset = []
for ovlp in candidates_dataset:
    if np.any(ovlp[:,:,1][:,:]==3):
        clean_subset.append(ovlp)
print(len(clean_subset),' examples')

resized_subs = ResizeImages(clean_subset)
subset_stacked = io.concatenate_images(resized_subs)
print(subset_stacked.shape)
print('Shuffling images')
np.random.shuffle(subset_stacked)
print(subset_stacked.shape)

kilo_byte = sys.getsizeof(clean_subset[0])/1024
print('Each image (grey+ground truth) weights:',kilo_byte,' kbytes')
print('Size of stacked images ',sys.getsizeof(subset_stacked)/1024/1024,' Mo')

np.random.shuffle(subset_stacked)

h5f = h5py.File('LowRes_90667_overlapping_pairs.h5', 'w')
h5f.create_dataset('dataset_1', data= subset_stacked, compression='gzip', compression_opts=9)
h5f.close()


