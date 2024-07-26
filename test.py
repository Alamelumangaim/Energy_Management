import pandas as pd
from ultralytics import YOLO
import cvzone
import cv2  
import numpy as np

model=YOLO('best.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture('video.mp4')
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
count=0
area1=[(1,288),(332,397),(307,310),(92,269)]
area2=[(751,382),(1075,426),(966,258),(768,308)]
area3 = [(468,472),(609,472),(603,388),(498,388)]
while True:    
    ret,frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    list = []
    list1 = []
    list2 = []
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        cx=int(x1+x2)//2
        cy=int(y1+y2)//2
        w,h=x2-x1,y2-y1
        result = cv2.pointPolygonTest(np.array(area1,np.int32),((cx,cy)),False)
        if result>=0:

#        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),-1)
            cvzone.cornerRect(frame,(x1,y1,w,h),3,2)
            cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
            cvzone.putTextRect(frame,f'person',(x1,y1),1,1)
            list.append(cx)
        result1 = cv2.pointPolygonTest(np.array(area2,np.int32),((cx,cy)),False)
        if result1>=0:

#        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),-1)
            cvzone.cornerRect(frame,(x1,y1,w,h),3,2)
            cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
            cvzone.putTextRect(frame,f'person',(x1,y1),1,1)
            list1.append(cx)
        result2 = cv2.pointPolygonTest(np.array(area3,np.int32),((cx,cy)),False)
        if result2>=0:

#        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),-1)
            cvzone.cornerRect(frame,(x1,y1,w,h),3,2)
            cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
            cvzone.putTextRect(frame,f'person',(x1,y1),1,1)
            list2.append(cx)
    cr1 = len(list)
    cr2 = len(list1)
    cr3 = len(list2)
    fan1 = len(list)>0
    fan2 = len(list1)>0
    fan3 = len(list2)>0
    print(fan1)
    print(fan2)
    print(fan3)
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,0,255),2)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,255,0),2)
    cv2.polylines(frame,[np.array(area3,np.int32)],True,(0,255,0),2)
    cvzone.putTextRect(frame,f'fan1: {cr1}',(50,30),2,2)
    cvzone.putTextRect(frame,f'fan2: {cr2}',(50,90),2,2)
    cvzone.putTextRect(frame,f'fan3: {cr3}',(50,150),2,2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()