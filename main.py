from os import system
import cv2 as cv
import numpy
from scipy import signal
import peakutils
from matplotlib import pyplot as plt
from math import sqrt, log, exp, pi
import time
from amplify_spatial_Gdown_temporal_ideal import amplify_spatial_Gdown_temporal_ideal
import matplotlib.animation as animation
import threading

bpm = []
bpmM = []
k = 0
n = 2000
now = time.time()
then = 0
vals = []
farr = []
proc = []
rate=0
means = medians = []
times = []
meanB = medianB = 0.0
frame = numpy.zeros((640,480,3),numpy.uint8)
timeF = 0
tim = []
dat = []
spec = []
periods = []

fig = plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)



def face_det(image):
    #image = adjust_gamma(image,2.2)
    face_cascade = cv.CascadeClassifier('C:\\opencv\\build\\share\\OpenCV\\haarcascades\\haarcascade_frontalface_alt.xml')
    eye_cascade = cv.CascadeClassifier('C:\\opencv\\build\\share\\OpenCV\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml')

    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    h_r,w_r,ch = image.shape
    roi= numpy.zeros((h_r,w_r,3), numpy.uint8)
    roi[:,:] = (255,255,255)
    i=0

    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x,y,w,h) in faces:
        i=1
        w1 = int(w*0.75)
        h1 = int(h*0.2)
        frame = cv.rectangle(image,(x+60,y),(x+w1,y+h1),(255,0,0),2)
        roi = image[y:y+h1, x+60:x+w1]

    cv.imshow("Face Track",image)
    cv.waitKey(10)

    return roi,i

def peakPulse(data):
    pPul = []

    for i in range(1,len(data)-1):
        if((data[i]==1)and(data[i-1]==0)):
            pPul.append(i)

    return pPul

def peaks(data):
    pVal = []
    
    for i in range(1,len(data)-1):
        if((data[i]>data[i-1])and(data[i]>data[i+1])):
            pVal.append(data[i])

    return pVal

def calcBPM(vals):
    val = 0
    for i in vals:
        if(i>(1.2*val)):
            val=i
        elif(i>(0.7*val)):
            val=i
    return val

def showHR(i):

    global means,medians,bpm,dat,spec,periods
    peakHR = peaks(bpm)
    print("\nMax Mean BPM = %.1f" % meanD(peakHR))

    res = calcBPM(bpm)
    print("Est. BPM = %.1f" % res)
    print(periods)

    ax1.clear()
    ax1.plot(bpm)
    ax1.set_ylabel('Heart Beat')
    ax1.grid(True)
    
    ax2.clear()
    ax2.plot(vals)
    ax2.set_ylabel('PPG Signal')
    ax2.grid(True)

    ax3.clear()
    ax3.plot(spec)
    ax3.set_ylabel('Frequency Spectrum')
    ax3.set_xlabel('Frames')
    ax3.grid(True)


def period(data):

    per = []

    for i in range(0,len(data)-1):
        per.append(data[i+1]-data[i])

    return per

def calc(ind):
    global n,k,q,then,now,vals,farr,times,means,medians,meanB,medianB,rate,bpm,dat,spec,proc,periods

    signal = proc[ind:ind+100]
    
    for l, frame in enumerate(signal):

        b,g,r = cv.split(frame)
        height,width,c = frame.shape 
        
        meanR=mean1(g)
        vals.append(meanR)
        if(meanR>=128):
            dat.append(1)
        else:
            dat.append(0)

        peakT = peakPulse(dat)
        periods = []
        for i in peakT:
            periods.append(times[ind+i])
        
        L = len(dat)
        processed = numpy.array(dat)

        if(L>10):
            fs = rate

            comp = numpy.fft.rfft(processed)
            phase = numpy.angle(comp)
            mag = numpy.abs(comp)
            freq = float(fs) / L * numpy.arange(len(mag))
            beats = 60. * freq
            index = numpy.where((beats > 30) & (beats < 160))
            peakMag = mag[index]
            index2 = numpy.argmax(peakMag)
            
            bpm.append(beats[index2])
            spec = mag


    

def mean1(img1):
    temp = numpy.mean(img1[:,:])
    return temp

def meanD(img1):
    temp = numpy.mean(img1[:])
    return temp

def mean2(img1,img2):
    temp = 0.0
    n = 0
    h,w = img1.shape
    for i in range(0,h):
            for j in range(0,w):
                temp = temp + img1.item(i,j) + img2.item(i,j)
                n=n+1
    temp = temp/float(2*n)
    return temp




def vidmag(m):
    global farr,proc,rate
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    mag = cv.VideoWriter('output.mp4',fourcc, 30, (120,96))
    frm = farr[m:m+50]
    
    for b, fr in enumerate(frm):
        fr = cv.resize(fr, (120,96), interpolation = cv.INTER_AREA)
        mag.write(fr)

    mag.release()

    amplify_spatial_Gdown_temporal_ideal("output.mp4","", 80,4,30.0/60.0,160.0/60.0,1.8,'rgb')

    vid = cv.VideoCapture("proc.mov")
    while(True):
            r,f=vid.read()
            if(r==True):
                    proc.append(f)
            else:
                    break
    vid.release()

def record():
    
    global n,rate,frame,farr,timeF,tim
    p = 0
    frames = []

    cap = cv.VideoCapture(0)

    while(len(farr)<n):
        rt,fr = cap.read()
        then = time.time()
        if (rt==False):
            continue
        if cv.waitKey(1) & 0xFF == 27:
            break

        que = threading.Thread(target=frames.insert, args=(0,fr))
        queT = threading.Thread(target=tim.insert, args=(0,then-now))
        
        que.start()
        queT.start()
        frame=frames.pop()
        timeF=tim.pop()
        p=p+1


        rate = int(p/(then-now))
        
    cap.release()
    print("FPS = %d" % rate)
    print("Capture Time = %d" % (then-now))
    #print(len(farr))
    

def featureExt():
   
    global n,k,rate,farr,frame,timeF,proc
    

    rec = threading.Thread(target=record)

    p = 0
    rec.start()
    cv.waitKey(10)

    while(True):
        evm = threading.Thread(target=vidmag, args=(k,))
        monitor = threading.Thread(target=calc, args=(p,))
        img=frame.copy()

        roi_color,i = face_det(img)

        if(i==0):
            print("No Subject.")
            continue
        
        if(len(farr)!=n):
            farr.append(roi_color)
            times.append(timeF)

        if((len(farr)%50)==0):          
            evm.start()
            k=k+50

        if(len(farr)==n):
            break

        if((len(proc)-p*100)==100):
            p = p+100
            monitor.start()
            

    





if __name__ == '__main__':
    
    run = threading.Thread(target=featureExt)
    run.start()
    ani = animation.FuncAnimation(fig, showHR, interval=1000)
    plt.show()
    
    cv.destroyAllWindows()
