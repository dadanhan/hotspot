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
        video = cv2.VideoWriter(outputname+".avi",fourcc, 2, (width,height))
        #video = cv2.VideoWriter(outputname+".avi",fourcc, 1, (2*width,height))
        #find peaks 
        numpeaks = 1000
        peakcoords = np.float32(f.peakfind(images[0],medsize,numpeaks)).reshape(-1,1,2)
        oldpeaks = peakcoords
        #make a frame for first frame
        framenow = np.array(images[0],dtype='float')
        framenow = framenow/np.mean(framenow)
        framenow *= 255.0/zmax
        framenow = cv2.medianBlur(np.array(framenow,dtype='uint8'),medsize)
        cf = cv2.applyColorMap(framenow,cv2.COLORMAP_JET)
        for n in peakcoords:
            for i,j in n: 
                cv2.circle(cf, (j,i), radius=3, color=(0,0,0), thickness=1, lineType=0, shift=0) 
        #write video
        video.write(cf)
        #frame-by-frame analysis
        for fn in range(1,len(images)):    
            framenow = np.array(images[fn],dtype='float')
            frameold = np.array(images[fn-1],dtype='float')
            #normalize frames with mean
            framenow = framenow/np.mean(framenow)
            framenow *= 255.0/zmax
            frameold = frameold/np.mean(frameold)
            frameold *= 255.0/zmax
            #median blur frame
            framenow = cv2.medianBlur(np.array(framenow,dtype='uint8'),medsize)
            frameold = cv2.medianBlur(np.array(frameold,dtype='uint8'),medsize)
            #framenow = np.array(ndi.median_filter(frame,medsize),dtype='uint8')
            #frameold = np.array(ndi.median_filter(frame,medsize),dtype='uint8')
            #outline frame
            #apply feature detection
            #(kps_now,des_now) = f.siftpic(framenow)
            (kps_now,des_now) = f.surfpic(framenow)
            #(kps_now,des_now) = f.orbpic(framenow)
            #(kps_old,des_old) = f.siftpic(frameold)
            (kps_old,des_old) = f.surfpic(frameold)
            #kps_old,des_old) = f.orbpic(frameold)
        #    #draw the image
        #    frame1 = cv2.drawKeypoints(framenow,kps_now,None,(255,255,255),flags=2)
        #    frame2 = cv2.drawKeypoints(framenow,kps_old,None,(255,255,255),flags=2)
            ##########################################FLANN based matching
            M,matchesMask,good,dst = f.flannmatch(frameold,kps_old,des_old,framenow,kps_now,des_now)
            ##########################################find peaks in frame
            #numpeaks = 100
            #peakcoords = f.peakfind(framenow,medsize,numpeaks)
            #print peakcoords
            ##########################################use homography to shift points on image by homography before pixel shift
            #print M
            #oldpeaks = cv2.perspectiveTransform(oldpeaks,M)
             #########################################find the average drift
            pixeldriftsx = []
            pixeldriftsy = []
            for m in good:
                #find new pixel co-ord - old pixel co-ord
                (xo,yo) = kps_old[m.queryIdx].pt
                (xn,yn) = kps_now[m.trainIdx].pt
                pixeldriftsx.append(xn-xo)
                pixeldriftsy.append(yn-yo)
            print "x: "+str(np.mean(pixeldriftsx))+" y: "+str(np.mean(pixeldriftsy))
            
            for cdx in range(0,len(oldpeaks)):
                oldpeaks[cdx][0][0] += np.median(pixeldriftsy) 
                oldpeaks[cdx][0][1] += np.median(pixeldriftsx)
                #print "x: "+str(cds[0][0])+" y: "+str(cds[0][1])
            ##########################################use homography to shift points on image
            #print M
            newpeaks = oldpeaks
            #newpeaks = cv2.perspectiveTransform(oldpeaks,M)
            oldpeaks = newpeaks
            ##########################################draw the good matches
            # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
            #                 singlePointColor = None,
            #                 matchesMask = matchesMask, # draw only inliers
            #                 flags = 2)         
            # #cv2.drawMatchesKnn expects list of lists as matches
            # image_matches = cv2.drawMatches(frameold,kps_old,framenow,kps_now,good, None,**draw_params)
            ##########################################
            #convert to colormap
            #cf = cv2.applyColorMap(image_matches,cv2.COLORMAP_JET)
            framenow = cv2.polylines(framenow,[np.int32(dst)],True,255,3, cv2.LINE_AA)
            cf = cv2.applyColorMap(framenow,cv2.COLORMAP_JET)
            
            for n in newpeaks:
                for i,j in n: 
                    cv2.circle(cf, (j,i), radius=3, color=(0,0,0), thickness=1, lineType=0, shift=0) 
            #cv2.putText(cf, "Old",(10,20),cv2.FONT_HERSHEY_TRIPLEX,0.5,(255,255,255))
            #cv2.putText(cf, "Now",(width+10,20),cv2.FONT_HERSHEY_TRIPLEX,0.5,(255,255,255))
            #write video
            video.write(cf)
                    
        video.release()
