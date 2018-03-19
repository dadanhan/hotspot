import functions as f
import numpy as np
import cv2
from scipy import ndimage as ndi

#all the video names and the zstack ranges
vidnames = ["040417E10_5_VH5_Het_pos1_tiffseries","04082016_VH5_Het_E10_5_pos2_tiffseries","13012017_E10_5_VH5Het_pos5_tiffseries","16122016_VH5Het_E10_5_pos5_tiffseries","26072016_VH5_Het_E10_5_pos4_croptime","28032017_VH5Het_E10_5_pos1_tiffseries","28032017_VH5Het_E10_5_pos6_tiffseries"]
zss = [range(0,10),range(0,13), range(0,12),range(0,13),range(0,12),range(0,11),range(0,11)]
medsizes = [5,11,5,5,11,5,5]
#medsizes = [11,21,11,11,21,11,11]

#for vnnum in range(0,len(vidnames)):
for vnnum in range(0,1):
    #assign z values
    zs = zss[vnnum]
    vidname = vidnames[vnnum]
    medsize = medsizes[vnnum]
    print vidname
    #print zs
    #for i in zs:
    for i in range(0,1):
        #zstack position
        zl = i+1
        #type of image
        cl = 1
        #import images
        images,outputname = f.readImages(vidname,zl,cl)
        images = np.array(images,dtype=float)
        time,width,height = images.shape
        print images.shape
        #find the maximum of this zstack image
        zmax = 0
        for ts in range(0,time):
            frame = images[ts]
            frame /= np.mean(frame)
            frame1 = np.array(ndi.median_filter(frame,size=medsize),dtype='float')
            if (np.max(frame1)>zmax):
                zmax = np.max(frame1)
        #make video
        print outputname
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #video = cv2.VideoWriter(outputname+"nothing.avi",fourcc, 2, (width,height))
        video = cv2.VideoWriter(outputname+"thresh.avi",fourcc, 2, (2*width,height))
        #frame-by-frame analysis
        for fn in range(0,len(images)):    
            framenow = np.array(images[fn],dtype='float')
            #normalize frames with mean
            framemean = np.mean(framenow)
            framenow = framenow/framemean
            framenow *= 255.0/zmax
            framenow = np.array(framenow,dtype='uint8')
            #blur frame
            n1 = 7
            #frameblur = cv2.GaussianBlur(framenow,(n1,n1),3)
            edit = cv2.medianBlur(framenow,n1)
            original = edit
            cf1 = cv2.applyColorMap(original,cv2.COLORMAP_JET)
            frameblur = edit
            #thresholds
            lowlimit = np.percentile(frameblur,95)
            ret,th1 = cv2.threshold(frameblur,lowlimit,255,cv2.THRESH_BINARY)
            frameblur[th1==0] = 0
            #erode
            n = 3
            kernel = np.ones((n,n),np.uint8)
            th1 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
            frameblur[th1==0] = 0
            # n = 3
            # kernel = np.ones((n,n),np.uint8)
            # th1 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)
            #find peaks
            coordinates = f.peakfind(frameblur,n,200)

            #create the video
            cf2 = cv2.applyColorMap(frameblur,cv2.COLORMAP_JET)
            for i,j in coordinates:
                cv2.circle(cf1, (j,i), radius=2, color=(0,0,0), thickness=1, lineType=0, shift=0) 
            cf = np.concatenate((cf1,cf2),axis=1)
            #write video
            video.write(cf)
        video.release()
