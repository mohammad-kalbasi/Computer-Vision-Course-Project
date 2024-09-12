# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 13:55:54 2020

@author: ASUS
"""
#%%
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import copy
cap = cv2.VideoCapture('Video1.avi')
cap.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
framenum = cap.get(cv2.CAP_PROP_POS_FRAMES) #finding number of frames
msec = cap.get(cv2.CAP_PROP_POS_MSEC)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
Frames = []
for i in range(int(framenum)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret,frame = cap.read()
    Frames.append(frame)#saving all frames
cap.set(cv2.CAP_PROP_POS_AVI_RATIO,0)
Background = np.median(Frames, axis=0).astype(dtype=np.uint8)# median in desiered axis
cv2.imshow('frame', Background)
cv2.imwrite('Background.jpg', Background)
cv2.waitKey(0)
gBackground = cv2.cvtColor(Background, cv2.COLOR_BGR2GRAY) #grey background
HSVBackground =  cv2.cvtColor(Background, cv2.COLOR_BGR2HSV)
cap.release()
cv2.destroyAllWindows()


#%% finding best background substracker
import numpy as np
import cv2

cap = cv2.VideoCapture("Video1.avi")
width =  cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

width = int(width)
height = int(height)
print( width, height)

  
# information to start saving a video file
ret, frame = cap.read()  # import image
ratio = 1.0  # resize ratio if needed 
image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
width2, height2, channels = image.shape
#video = cv2.VideoWriter('traffic_counter.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (height2, width2), 1)
f_counter = 0
while True:
    ret, frame = cap.read()  # import image
    if not ret: #if vid finish repeat
        break
    if ret:  # if there is a frame continue with code
        image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
        cv2.imshow("image", image) #@
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray
        cv2.imshow("gray", gray) #@
        substracted = cv2.absdiff(gBackground, gray)  # uses the background subtraction
        # applies different thresholds to substracted to try and isolate cars
        # just have to keep playing around with settings until cars are easily identifiable
        retvalbin, bins = cv2.threshold(substracted, 30, 255, cv2.THRESH_BINARY)
        cv2.imshow("substracted", bins) #@
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # kernel to apply to the morphology
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
        kernel4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # kernel to apply to the morphology
        dilation = cv2.dilate(bins, kernel1)
        cv2.imshow("dilation", dilation) #@
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel2)
        cv2.imshow("closing", closing) #@
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel3)
        cv2.imshow("opening", opening) #@
        erosion = cv2.erode(opening, kernel4)
        cv2.imshow("erosion", erosion) #@

        retvalbin, augmented = cv2.threshold(erosion, 30, 255, cv2.THRESH_BINARY)  # removes the shadows
        cv2.imshow("augmented", augmented) #@
        if f_counter == 300:
            cv2.imwrite('augmented.jpg', augmented)
            cv2.imwrite('bins.jpg', bins)

        f_counter = f_counter+1

    key = cv2.waitKey(20)
    if key == 27:
       break

cap.release()
cv2.destroyAllWindows()





#%% kalman filter

class KalmanFilter(object):

    def __init__(self):
        self.dt = 0.005
        self.A = np.array([[1, 0], [0, 1]])
        self.u = np.zeros((2, 1))
        self.b = np.array([[0], [255]])
        self.P = np.diag((3.0, 3.0))
        self.F = np.array([[1.0, self.dt], [0.0, 1.0]])
        self.Q = np.eye(self.u.shape[0])
        self.R = np.eye(self.b.shape[0])
        self.lastResult = np.array([[0], [255]])

    def predict(self):
        self.u = np.round(np.dot(self.F, self.u))
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        self.lastResult = self.u
        return self.u

    def correct(self, b, flag):
        if not flag:
            self.b = self.lastResult
        else:
            self.b = b
        C = np.dot(self.A, np.dot(self.P, self.A.T)) + self.R
        K = np.dot(self.P, np.dot(self.A.T, np.linalg.inv(C)))
        self.u = np.round(self.u + np.dot(K, (self.b - np.dot(self.A, self.u))))
        self.P = self.P - np.dot(K, np.dot(C, K.T))
        self.lastResult = self.u
        return self.u
#%% first class getting tracks
class Tracked(object):
    def __init__(self, pred_cent, current_ID):
        self.KF = KalmanFilter()
        self.ID = current_ID
        self.pred_cent = np.asarray(pred_cent)
        self.missing = 0
        self.frame = []
        self.first_frame = []
        self.velocity = 0
#%% second class is used for finding related tracks
#%% run this for main parts+ oprionals
class Tracking(object):
    def __init__(self, max_dist, max_missing,min_presence):
        
         
        self.max_dist = max_dist
        self.max_missing = max_missing
        self.min_presence = min_presence
        self.current_ID = 0
        self.tracking = []
        self.track_ended = [] # we save final tracks in this
    
    
    def Update_Track(self, frame, contours, centers, t_counter,fps):
        # some explanation
          """Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Calculate cost using sum of square distance
              between predicted vs detected centroids
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Identify tracks with no assignment, if any
            - If tracks are not detected for long time, remove them
            - Now look for un_assigned detects
            - Start new tracks
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            detections: detected centroids of object to be tracked
        Return:
            None
        """
          if len(self.tracking) == 0:
            for i in range(len(centers)):
                # initializing tracks
                track = Tracked(centers[i], self.current_ID)
                self.current_ID += 1
                self.tracking.append(track)
          N_track_num = len(self.tracking)
          M_cent_num = len(centers)
          cost_matrix = np.zeros(shape = (N_track_num,M_cent_num))
          for i in range(N_track_num):
            for j in range(M_cent_num):
                try:
                    diff = self.tracking[i].pred_cent - centers[j]
                    cost_matrix[i][j] = np.sqrt(diff[0][0] * diff[0][0] +
                                                diff[1][0] * diff[1][0])
                except:
                    pass
          cost_matrix = 0.5*cost_matrix # average cost matrix
          # Using Hungarian Algorithm assign the correct detected measurements
          # to predicted tracks
          assign = []
          for i in range(N_track_num):
             assign.append(-1)
          row_ind, col_ind = linear_sum_assignment(cost_matrix)
          
          for i in range(len(row_ind)):
            assign[row_ind[i]] = col_ind[i]
          # Identify tracks with no assignment, if any
           # Identify tracks with no assignment, if any
          un_assign = []

          for i in range(len(assign)):
              if (assign[i] != -1):
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                    if (cost_matrix[i][assign[i]] > self.max_dist):
                        assign[i] = -1
                        pass
              else:
                  self.tracking[i].missing = self.tracking[i].missing + 1
          deleting = []
          for i in range (len(self.tracking)):
              if self.tracking[i].missing > self.max_missing:
                  deleting.append(i)
                  if len(self.tracking[i].frame) > self.min_presence:
                      self.track_ended.append(self.tracking[i])
          del_counter = 0
          if len(deleting) > 0:
              for i in deleting:
                  if i < len(self.tracking):
                      del self.tracking[i-del_counter]
                      del assign[i-del_counter]
                      del_counter = del_counter+1
                  else:
                       print("ERROR: id is greater than length of tracks")
                       
          for i in range (M_cent_num):
              if i not in assign:
                  un_assign.append(i)
          # Start new tracks
          if len( un_assign) > 0:
              for i in range (len(un_assign)):
                  track = Tracked(centers[un_assign[i]], self.current_ID)
                  self.tracking.append(track)
                  self.current_ID = self.current_ID +1
          for i in range (len(assign)):
              self.tracking[i].KF.predict()
              if assign[i] != -1:
                  nex_predict_cnt = self.tracking[i].KF.correct(centers[assign[i]],1)
                  inital_val_const = 10 # we can change it manualy
                  if len(self.tracking[i].frame) == inital_val_const:
                      self.tracking[i].first_frame = nex_predict_cnt
                  # we detected it so we should say we see it in this frame!
                  self.tracking[i].missing = 0
                  # calculating reletive velocity
                  velocity_n = np.sqrt((self.tracking[i].pred_cent[0,0] - nex_predict_cnt[0,0])**2 + (self.tracking[i].pred_cent[1,0] - nex_predict_cnt[1,0])**2)
                  up_v_thresh = 40
                  velocity_n = np.min([up_v_thresh,velocity_n])
                  damping_ratio = 100
                  self.tracking[i].velocity = (self.tracking[i].velocity+damping_ratio*velocity_n)/(1+damping_ratio)
                  # now we don't need previous position of center we change it
                  self.tracking[i].pred_cent = nex_predict_cnt
                  mask = np.zeros(frame.shape)
                  mask = mask.astype(dtype = np.uint8)
                  cv2.drawContours(mask, [contours[assign[i]]], -1, (255, 255, 255), cv2.FILLED)
                  # for correct sumation we need it to be int
                  mask = mask.astype(dtype = np.int)
                  wheight = (np.sum(mask))/(255*3)
                  mask = mask.astype(dtype = np.uint8)

                  seprated = np.where(mask,frame,0)
                                    #again we need some calculation
                  #it's BGR
                  seprated = seprated.astype(dtype = np.int)
                  b_sum = np.sum(seprated[:,:,0])/wheight
                  g_sum = np.sum(seprated[:,:,1])/wheight
                  r_sum = np.sum(seprated[:,:,2])/wheight
                  colors_mean = np.array([b_sum,g_sum,r_sum])
                  seprated = seprated.astype(dtype = np.uint8)
                  cv2.rectangle(seprated,(centers[assign[i]][0,0],centers[assign[i]][1,0]),(centers[assign[i]][0,0]+13,centers[assign[i]][1,0]-13),colors_mean,-1)
                  cv2.putText(seprated,str(int(t_counter/(fps*60))) + ":"+ str(int(np.remainder(t_counter/fps, 60))),(centers[assign[i]][0,0],centers[assign[i]][1,0]),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,255,255),1)
                  self.tracking[i].velocity = float("{:.2f}".format(self.tracking[i].velocity))
                  cv2.putText(seprated,'relative v = ' + str((self.tracking[i].velocity)),(centers[assign[i]][0,0],centers[assign[i]][1,0]+13),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,255,255),1)
                  self.tracking[i].frame.append(seprated)
              else:
                  self.tracking[i].pred_cent = self.tracking[i].KF.correct(np.array([[0], [0]]), 0)
              self.tracking[i].KF.lastResult = self.tracking[i].pred_cent
             
    def ending(self):
        for i in range (len(self.tracking)):
            if (len(self.tracking[i].frame)) > self.min_presence:
                self.track_ended.append(self.tracking[i])
                 
#%% run this part if you just want main part
#important note: please change name to Tracking for proper answer
class Tracking2(object):
    def __init__(self, max_dist, max_missing,min_presence):
        
         
        self.max_dist = max_dist
        self.max_missing = max_missing
        self.min_presence = min_presence
        self.current_ID = 0
        self.tracking = []
        self.track_ended = [] # we save final tracks in this
    
    
    def Update_Track(self, frame, contours, centers, t_counter,fps):
        # some explanation
          """Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Calculate cost using sum of square distance
              between predicted vs detected centroids
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Identify tracks with no assignment, if any
            - If tracks are not detected for long time, remove them
            - Now look for un_assigned detects
            - Start new tracks
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            detections: detected centroids of object to be tracked
        Return:
            None
        """
          if len(self.tracking) == 0:
            for i in range(len(centers)):
                # initializing tracks
                track = Tracked(centers[i], self.current_ID)
                self.current_ID += 1
                self.tracking.append(track)
          N_track_num = len(self.tracking)
          M_cent_num = len(centers)
          cost_matrix = np.zeros(shape = (N_track_num,M_cent_num))
          for i in range(N_track_num):
            for j in range(M_cent_num):
                try:
                    diff = self.tracking[i].pred_cent - centers[j]
                    cost_matrix[i][j] = np.sqrt(diff[0][0] * diff[0][0] +
                                                diff[1][0] * diff[1][0])
                except:
                    pass
          cost_matrix = 0.5*cost_matrix # average cost matrix
          # Using Hungarian Algorithm assign the correct detected measurements
          # to predicted tracks
          assign = []
          for i in range(N_track_num):
             assign.append(-1)
          row_ind, col_ind = linear_sum_assignment(cost_matrix)
          
          for i in range(len(row_ind)):
            assign[row_ind[i]] = col_ind[i]
          # Identify tracks with no assignment, if any
           # Identify tracks with no assignment, if any
          un_assign = []

          for i in range(len(assign)):
              if (assign[i] != -1):
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                    if (cost_matrix[i][assign[i]] > self.max_dist):
                        assign[i] = -1
                        pass
              else:
                  self.tracking[i].missing = self.tracking[i].missing + 1
          deleting = []
          for i in range (len(self.tracking)):
              if self.tracking[i].missing > self.max_missing:
                  deleting.append(i)
                  if len(self.tracking[i].frame) > self.min_presence:
                      self.track_ended.append(self.tracking[i])
          del_counter = 0
          if len(deleting) > 0:
              for i in deleting:
                  if i < len(self.tracking):
                      del self.tracking[i-del_counter]
                      del assign[i-del_counter]
                      del_counter = del_counter+1
                  else:
                       print("ERROR: id is greater than length of tracks")
                       
          for i in range (M_cent_num):
              if i not in assign:
                  un_assign.append(i)
          # Start new tracks
          if len( un_assign) > 0:
              for i in range (len(un_assign)):
                  track = Tracked(centers[un_assign[i]], self.current_ID)
                  self.tracking.append(track)
                  self.current_ID = self.current_ID +1
          for i in range (len(assign)):
              self.tracking[i].KF.predict()
              if assign[i] != -1:
                  nex_predict_cnt = self.tracking[i].KF.correct(centers[assign[i]],1)
                  inital_val_const = 10
                  if len(self.tracking[i].frame) == inital_val_const:
                      self.tracking[i].first_frame = nex_predict_cnt
                  # we detected it so we should say we see it in this frame!
                  self.tracking[i].missing = 0
                  # now we don't need previous position of center we change it
                  self.tracking[i].pred_cent = nex_predict_cnt
                  mask = np.zeros(frame.shape)
                  mask = mask.astype(dtype = np.uint8)
                  cv2.drawContours(mask, [contours[assign[i]]], -1, (255, 255, 255), cv2.FILLED)
                  
                  seprated = np.where(mask,frame,0)
                  cv2.putText(seprated,str(int(t_counter/(fps*60))) + ":"+ str(int(np.remainder(t_counter/fps, 60))),(centers[assign[i]][0,0],centers[assign[i]][1,0]),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,255,255),1)

                  self.tracking[i].frame.append(seprated)
              else:
                  self.tracking[i].pred_cent = self.tracking[i].KF.correct(np.array([[0], [0]]), 0)
              self.tracking[i].KF.lastResult = self.tracking[i].pred_cent
             
    def ending(self):
        for i in range (len(self.tracking)):
            if (len(self.tracking[i].frame)) > self.min_presence:
                self.track_ended.append(self.tracking[i])
                  
#%% seprating counters and making tubes

cap = cv2.VideoCapture("Video2.avi")
width =  cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

width = int(width)
height = int(height)
print( width, height)

  
# information to start saving a video file
ret, frame = cap.read()  # import image
ratio = 1.0  # resize ratio if needed 
t_counter = 0
tubes = Tracking(250, 25, 100)

while True:
    ret, frame = cap.read()  # import image
    if not ret: #if vid finish repeat
        tubes.ending()
        break
    if ret:  # if there is a frame continue with code
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converts image to gray
        substracted = cv2.absdiff(gBackground, gray)  # uses the background subtraction
        # applies different thresholds to substracted to try and isolate cars
        # just have to keep playing around with settings until cars are easily identifiable
        _, bins = cv2.threshold(substracted, 30, 255, cv2.THRESH_BINARY)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # kernel to apply to the morphology
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
        kernel4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # kernel to apply to the morphology
        dilation = cv2.dilate(bins, kernel1)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel3)
        erosion = cv2.erode(opening, kernel4)

        _, augmented = cv2.threshold(erosion, 30, 255, cv2.THRESH_BINARY)  # removes the shadows
        Contours = []
        Centers = []
        contours, hierarchy = cv2.findContours(augmented,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        minarea = 50
        # max area for contours, can be quite large for buses

        maxarea = 50000
        for i in range(len(contours)):
            if hierarchy[0, i, 3] == -1:  # using hierarchy to only count parent contours (contours not within others)
                area = cv2.contourArea(contours[i])  # area of contour
                if minarea < area < maxarea:  # area threshold for contour
                    # calculating centroids of contours
                    cnt = contours[i]
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    Contours.append(contours[i])
                    Centers.append(np.round(np.array([[cx],[cy]]))) # we should round it to use it
        if len(Centers) > 0:# we at least detected something to start our work!
            tubes.Update_Track(frame, Contours, Centers, t_counter,30)
        
        t_counter = t_counter+1

for i in range(len(tubes.track_ended)):
    if i == 0:
        max_lenght = len(tubes.track_ended[0].frame)
    else:
        if len(tubes.track_ended[i].frame) > max_lenght:
           max_lenght =  len(tubes.track_ended[i].frame)
start_frame = np.zeros(len(tubes.track_ended))
for i in range(len(tubes.track_ended)):
    length_diffrence = max_lenght - len(tubes.track_ended[i].frame)
    start_frame[i] = np.round(np.random.random(1) *length_diffrence)
start_frame = start_frame.astype(np.uint)



out = cv2.VideoWriter('output1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), int(frame_rate),(width,height) ) # creating codec of video
for i in range(max_lenght):
    current_frame = copy.deepcopy(Background) # in each frame we need to use background without change the original one
    for tube_num in range(len(tubes.track_ended)):
        if ((start_frame[tube_num] < i) and (i < start_frame[tube_num] + len(tubes.track_ended[tube_num].frame))):
            mask = cv2.cvtColor(tubes.track_ended[tube_num].frame[i - start_frame[tube_num]],cv2.COLOR_BGR2GRAY)
            temp = np.zeros(Background.shape)
            temp[:,:,0] = mask
            temp[:,:,1] = mask
            temp[:,:,2] = mask
            current_frame = np.where(temp != 0,tubes.track_ended[tube_num].frame[i - start_frame[tube_num]],current_frame )
    out.write(current_frame)
out.release()
cap.release()
cv2.destroyAllWindows()
#%% we need a function to calculate our phase
from math import degrees
from math import atan

def phase_cal(x,y):
    if x > 0:
        phase = degrees(atan(y/x))
    elif x < 0:
         if y> 0:
             phase = degrees(atan(y/x)) + 180
         elif y < 0:
             phase=  degrees(atan(y/x)) - 180
         else:
             phase = 0
    else: 
        phase = 0
        
    return phase
    
#%% for creating track video1
from math import sin,cos,radians
cap = cv2.VideoCapture("Video1.avi")
width =  cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

width = int(width)
height = int(height)
print( width, height)

  
# information to start saving a video file
ret, frame = cap.read()  # import image

#video = cv2.VideoWriter('traffic_counter.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (height2, width2), 1)
t_counter = 0
tubes = Tracking(25, 5, 50)

while True:
    ret, frame = cap.read()  # import image
    if not ret: #if vid finish repeat
        tubes.ending()
        break
    if ret:  # if there is a frame continue with code
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converts image to gray
        substracted = cv2.absdiff(gBackground, gray)  # uses the background subtraction
        # applies different thresholds to substracted to try and isolate cars
        # just have to keep playing around with settings until cars are easily identifiable
        _, bins = cv2.threshold(substracted, 30, 255, cv2.THRESH_BINARY)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # kernel to apply to the morphology
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
        kernel4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # kernel to apply to the morphology
        dilation = cv2.dilate(bins, kernel1)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel3)
        erosion = cv2.erode(opening, kernel4)

        _, augmented = cv2.threshold(erosion, 30, 255, cv2.THRESH_BINARY)  # removes the shadows
        Contours = []
        Centers = []
        contours, hierarchy = cv2.findContours(augmented,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        minarea = 50
        # max area for contours, can be quite large for buses

        maxarea = 50000
        for i in range(len(contours)):
            if hierarchy[0, i, 3] == -1:  # using hierarchy to only count parent contours (contours not within others)
                area = cv2.contourArea(contours[i])  # area of contour
                if minarea < area < maxarea:  # area threshold for contour
                    # calculating centroids of contours
                    cnt = contours[i]
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    Contours.append(contours[i])
                    Centers.append(np.round(np.array([[cx],[cy]]))) # we should round it to use it
        if len(Centers) > 0:# we at least detected something to start our work!
            tubes.Update_Track(frame, Contours, Centers, t_counter,30)
        
        t_counter = t_counter+1
cap.release()
length_v = np.zeros(len(tubes.track_ended))
phase = np.zeros(len(tubes.track_ended))
for i in range(len(tubes.track_ended)):
    x_vector = float(tubes.track_ended[i].pred_cent[1,0] - tubes.track_ended[i].first_frame[1,0])
    y_vector = float(tubes.track_ended[i].pred_cent[0,0] - tubes.track_ended[i].first_frame[0,0])
    phase[i] = int(phase_cal(x_vector,y_vector))
    length_v[i] = int(np.linalg.norm([x_vector,y_vector]))
selected = []
mag_th = 100 # change it manualy it help us to find bigest and best tracks
chosen_end = []
for i in range(len(tubes.track_ended)):
    if (length_v[i] > mag_th) and  (phase[i] != 0):
        selected.append(i)
        chosen_end.append(tubes.track_ended[i])

chosen_len = length_v[selected]
chosen_ph = phase[selected]
chosen_ph = np.float32(chosen_ph)

_, tags, centers = cv2.kmeans(chosen_ph, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
maxlen0 = np.max(chosen_len[tags[:,0] == 0])
maxlen1 = np.max(chosen_len[tags[:,0] == 1])
pop0 = len(chosen_len[tags[:,0] == 0])
pop1 = len(chosen_len[tags[:,0] == 1])
maxlen0 = 0
maxlen1 = 0
for i in range(len(chosen_end)):
    if tags[i,0] == 0:
        if chosen_len[i] > maxlen0:
            maxlen0 = chosen_len[i]
            maxlen0_ind = i
    if tags[i,0] == 1:
        if chosen_len[i] > maxlen1:
            maxlen1 = chosen_len[i]
            maxlen1_ind = i
firstpoint0 = tubes.track_ended[maxlen0_ind].first_frame
lastpoint0 = np.round(np.array([[firstpoint0[0,0] + maxlen0 * sin(radians(centers[0,0]))], [firstpoint0[1,0] + maxlen0* cos(radians(centers[0,0]))]]))

firstpoint1 = tubes.track_ended[maxlen1_ind-1].first_frame
firstpoint1_temp = tubes.track_ended[maxlen1_ind-1].first_frame
lastpoint1 = np.round(np.array([[firstpoint1[0,0] + maxlen1 * sin(radians(centers[1,0]))], [firstpoint1[1,0] + maxlen1* cos(radians(centers[1,0]))]]))
lined_back = copy.deepcopy(Background)
cv2.line(lined_back, (int(firstpoint0[0,0]), int(firstpoint0[1,0])), (int(lastpoint0[0,0]), int(lastpoint0[1,0])), (255, 255, 255), 7)
cv2.line(lined_back, (int(firstpoint1_temp[0,0]), int(firstpoint1_temp[1,0])), (int(lastpoint1[0,0]), int(lastpoint1[1,0])), (0, 0, 0), 7)
estimated_num = 3*len(tubes.track_ended)/len(chosen_end)
cv2.putText(lined_back, str(int(pop0 * estimated_num)), (int(firstpoint0[0,0]), int(firstpoint0[1,0])), cv2.FONT_HERSHEY_SCRIPT_COMPLEX , 0.5, (255, 255, 255), 1)       
cv2.putText(lined_back, str(int(pop1 * estimated_num)), (int(firstpoint1_temp[0,0]), int(firstpoint1_temp[1,0])), cv2.FONT_HERSHEY_SCRIPT_COMPLEX , 0.5, (255, 255, 255), 1)    
cv2.imshow('background', lined_back)
cv2.waitKey(0)
cv2.imwrite('line_detect.jpg',lined_back)
           
    
cv2.destroyAllWindows()


#%% for creating track video2
from math import sin,cos,radians
cap = cv2.VideoCapture("Video2.avi")
width =  cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

width = int(width)
height = int(height)
print( width, height)

  
# information to start saving a video file
ret, frame = cap.read()  # import image

#video = cv2.VideoWriter('traffic_counter.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (height2, width2), 1)
t_counter = 0
tubes = Tracking(25, 5, 50)

while True:
    ret, frame = cap.read()  # import image
    if not ret: #if vid finish repeat
        tubes.ending()
        break
    if ret:  # if there is a frame continue with code
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converts image to gray
        substracted = cv2.absdiff(gBackground, gray)  # uses the background subtraction
        # applies different thresholds to substracted to try and isolate cars
        # just have to keep playing around with settings until cars are easily identifiable
        _, bins = cv2.threshold(substracted, 30, 255, cv2.THRESH_BINARY)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # kernel to apply to the morphology
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
        kernel4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # kernel to apply to the morphology
        dilation = cv2.dilate(bins, kernel1)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel3)
        erosion = cv2.erode(opening, kernel4)

        _, augmented = cv2.threshold(erosion, 30, 255, cv2.THRESH_BINARY)  # removes the shadows
        Contours = []
        Centers = []
        contours, hierarchy = cv2.findContours(augmented,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        minarea = 50
        # max area for contours, can be quite large for buses

        maxarea = 50000
        for i in range(len(contours)):
            if hierarchy[0, i, 3] == -1:  # using hierarchy to only count parent contours (contours not within others)
                area = cv2.contourArea(contours[i])  # area of contour
                if minarea < area < maxarea:  # area threshold for contour
                    # calculating centroids of contours
                    cnt = contours[i]
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    Contours.append(contours[i])
                    Centers.append(np.round(np.array([[cx],[cy]]))) # we should round it to use it
        if len(Centers) > 0:# we at least detected something to start our work!
            tubes.Update_Track(frame, Contours, Centers, t_counter,30)
        
        t_counter = t_counter+1
cap.release()
length_v = np.zeros(len(tubes.track_ended))
phase = np.zeros(len(tubes.track_ended))
for i in range(len(tubes.track_ended)):
    x_vector = float(tubes.track_ended[i].pred_cent[1,0] - tubes.track_ended[i].first_frame[1,0])
    y_vector = float(tubes.track_ended[i].pred_cent[0,0] - tubes.track_ended[i].first_frame[0,0])
    phase[i] = int(phase_cal(x_vector,y_vector))
    length_v[i] = int(np.linalg.norm([x_vector,y_vector]))
selected = []
mag_th = 200 # change it manualy it help us to find bigest and best tracks
chosen_end = []
for i in range(len(tubes.track_ended)):
    if (length_v[i] > mag_th) and  (phase[i] != 0):
        selected.append(i)
        chosen_end.append(tubes.track_ended[i])

chosen_len = length_v[selected]
chosen_ph = phase[selected]
chosen_ph = np.float32(chosen_ph)

_, tags, centers = cv2.kmeans(chosen_ph, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
maxlen0 = np.max(chosen_len[tags[:,0] == 0])
maxlen1 = np.max(chosen_len[tags[:,0] == 1])
pop0 = len(chosen_len[tags[:,0] == 0])
pop1 = len(chosen_len[tags[:,0] == 1])
maxlen0 = 0
maxlen1 = 0
for i in range(len(chosen_end)):
    if tags[i,0] == 0:
        if chosen_len[i] > maxlen0:
            maxlen0 = chosen_len[i]
            maxlen0_ind = i
    if tags[i,0] == 1:
        if chosen_len[i] > maxlen1:
            maxlen1 = chosen_len[i]
            maxlen1_ind = i
firstpoint0 = tubes.track_ended[maxlen0_ind].first_frame
lastpoint0 = np.round(np.array([[firstpoint0[0,0] + maxlen0 * sin(radians(centers[0,0]))], [firstpoint0[1,0] + maxlen0* cos(radians(centers[0,0]))]]))
firstpoint1 = tubes.track_ended[maxlen1_ind-1].first_frame
firstpoint1_temp = tubes.track_ended[maxlen1_ind-1].first_frame
lastpoint1 = np.round(np.array([[firstpoint1[0,0] + maxlen1 * sin(radians(centers[1,0]))], [firstpoint1[1,0] + maxlen1* cos(radians(centers[1,0]))]]))
lined_back = copy.deepcopy(Background)
cv2.line(lined_back, (int(firstpoint0[0,0]), int(firstpoint0[1,0])), (int(lastpoint0[0,0]), int(lastpoint0[1,0])), (255, 255, 255), 7)
cv2.line(lined_back, (int(firstpoint1_temp[0,0]), int(firstpoint1_temp[1,0])), (int(lastpoint1[0,0]), int(lastpoint1[1,0])), (0, 0, 0), 7)
estimated_num = 2*len(tubes.track_ended)/len(chosen_end)
cv2.putText(lined_back, str(int(pop0 * estimated_num)), (int(firstpoint0[0,0]), int(firstpoint0[1,0])), cv2.FONT_HERSHEY_SCRIPT_COMPLEX , 0.5, (255, 255, 255), 1)       
cv2.putText(lined_back, str(int(pop1 * estimated_num)), (int(firstpoint1_temp[0,0]), int(firstpoint1_temp[1,0])), cv2.FONT_HERSHEY_SCRIPT_COMPLEX , 0.5, (255, 255, 255), 1)    
cv2.imshow('background', lined_back)
cv2.waitKey(0)
cv2.imwrite('line_detect.jpg',lined_back)
           
    
cv2.destroyAllWindows()


    
    