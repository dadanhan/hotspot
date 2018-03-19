import cv2
import os
import numpy as np
from skimage.feature import peak_local_max

#function for reading in image files
def readImages(pathname,zlabel,clabel):
    overpath = "/home/dani/Dropbox (The University of Manchester)/geneOscillation/CerysVids/"
    images = []
    filenames = []
    #print overpath+pathname+"/"
    #loop through every file in directory and find the ones that match condition
    for filename in os.listdir(overpath+pathname+"/"):
        if filename.endswith("_z"+str(zlabel).zfill(3)+"_c"+str(clabel).zfill(3)+".tif"):
            images.append(np.array(cv2.imread(overpath+pathname+"/"+filename,0),dtype=float))
            filenames.append(filename)
    #sort the filenames
    filenames,images=zip(*sorted(zip(filenames, images)))
    outputname = pathname[0:6]+"_z"+str(zlabel)+"_c"+str(clabel)
    return[images,outputname]

#function for applying sift with no params
def siftpic_noparams(gray):
    sift = cv2.xfeatures2d.SIFT_create()
    (kps, des) = sift.detectAndCompute(gray, None)
    print("#   SIFTNP kps: {}, descriptors: {}".format(len(kps), des.shape))
    return[kps,des]

#function for applying sift
def siftpic(gray):
    nfeatures = 500 #The number of best features to retain. The features are ranked by their scores (measured in SIFT algorithm as the local contrast)
    nOctaveLayers = 10 #The number of layers in each octave. 3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.
    contrastThreshold = 0.01 #The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
    edgeThreshold = 100 #The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).
    sigma = 5 #The sigma of the Gaussian applied to the input image at the octave #0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number. 
    sift = cv2.xfeatures2d.SIFT_create(nfeatures,nOctaveLayers,contrastThreshold,edgeThreshold,sigma)
    (kps, des) = sift.detectAndCompute(gray, None)
    print("#   SIFT kps: {}, descriptors: {}".format(len(kps), des.shape))
    return[kps,des]
    
#function for applying surf
def surfpic(gray):
    #apply surf to image
    hessianThreshold = 300 #Threshold for hessian keypoint detector used in SURF. 
    nOctaves = 10 #Number of pyramid octaves the keypoint detector will use. 
    nOctaveLayers = 3 #Number of octave layers within each octave. 
    extended = False #Extended descriptor flag (true - use extended 128-element descriptors; false - use 64-element descriptors). 
    upright = True #Up-right or rotated features flag (true - do not compute orientation of features; false - compute orientation).
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold,nOctaves,nOctaveLayers,extended,upright)
    (kps, des) = surf.detectAndCompute(gray, None)
    print("SURF kps: {}, descriptors: {}".format(len(kps), des.shape))
    return[kps,des]
    
#function for applying orb
def orbpic(gray):
    #Initiate detector
    orb = cv2.ORB_create()
    #find keypoints and descriptors with ORB
    kps,des = orb.detectAndCompute(gray,None)
    print("ORB kps: {}, descriptors: {}".format(len(kps), des.shape))
    return[kps,des]

#function for applying FLANN based matching
def flannmatch(frameold,kps_old,des_old,framenow,kps_now,des_now):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 500)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_old,des_now,k=2)

    #apply ratio test as in Lowe 2004
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    
    # for m in good:
    #     print kps_now[m.queryIdx].pt

    #Now we set a condition that atleast 10 matches (defined by MIN_MATCH_COUNT) are to be there to find the object. Otherwise simply show a message saying not enough matches are present.
    #If enough matches are found, we extract the locations of matched keypoints in both the images. They are passed to find the perpective transformation. Once we get this 3x3 transformation matrix, we use it to transform the corners of queryImage to corresponding points in trainImage. Then we draw it.
    MIN_MATCH_COUNT = 10
   #if len(good)>MIN_MATCH_COUNT:
    print len(good)
    src_pts = np.float32([ kps_old[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kps_now[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = frameold.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    #framenow = cv2.polylines(framenow,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    # else:
    #     print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    #     matchesMask = None
    return M,matchesMask, good, dst

#function for finding peaks
def peakfind(frame,medsize,numpeaks):
    #convert to uint8
    frame = np.array(frame,dtype='uint8')
    ##########################################morphological transformations
    #otsu binarization  
    ret, thresh = cv2.threshold(frame,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # noise removal
    n = 2
    kernel = np.ones((n,n),np.uint8)
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    n = 4
    kernel = np.ones((n,n),np.uint8)
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations = 4)
    n = 3
    kernel = np.ones((n,n),np.uint8)
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 6)
    # sure background area
    #thresh = cv2.erode(thresh,kernel,iterations=3)
    mask = np.array(frame,dtype='float')
    mask[thresh == 1] = 0
    ##########################################find peaks
    # Comparison between image_max and im to find the coordinates of local maxima
    #absolute threshold for peak finder
    absthresh = np.mean(np.array(frame,dtype='float'))
    #find peaks
    coordinates = peak_local_max(mask,min_distance=medsize,threshold_abs=absthresh,num_peaks=numpeaks)
    #coordinates = peak_local_max(frame, min_distance=medsize,threshold_abs=absthresh,threshold_rel=None,exclude_border=True,indices=True,num_peaks=50,footprint=mask,labels=None,num_peaks_per_label=np.inf)

    return coordinates
