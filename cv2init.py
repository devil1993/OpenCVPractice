import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==================================================================================================================STEP1============
img = cv2.imread('p.jpg',cv2.IMREAD_GRAYSCALE)
print(img)
cv2.imshow('Image00', img)
cv2.waitKey()

cv2.destroyAllWindows()


# plt.imshow(img, cmap = 'gray' , interpolation = 'bicubic')
# plt.show()


# cv2.imwrite('name.jpg',img)

# ==================================================================================================================STEP2============

# cap = cv2.VideoCapture(0)
# # cap = cv2.VideoCapture('output.avi')
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

# while True:
# 	ret, frame = cap.read()
# 	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

# 	cv2.imshow('frame',frame)
# 	cv2.imshow('gray',gray)

# 	out.write(frame)
	
# 	if(cv2.waitKey(1) & 0xFF == ord('q')):
# 		break

# cap.release()
# out.release()
# cv2.destroyAllWindows()

# ===================================================================================================================STEP3============== 

# img = cv2.imread('p.jpg',cv2.IMREAD_COLOR)

# cv2.line(img,(0,0),(200,200),(255,255,255),15)
# cv2.rectangle(img,(15,15),(356,400),(0,255,0),5)
# cv2.circle(img,(459,450),55,(0,0,0),2)
# pts = np.array([[12,23],[44,51],[85,23],[23,222]],np.int32)
# # pts = pts.reshape()
# cv2.polylines(img,[pts],True,(255,0,0),3)

# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img,'OpenCV Tuts!',(10,500), font, 6, (200,255,155), 13, cv2.LINE_AA)

# cv2.imshow('image',img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# ====================================================================================================================STEP4===========


# img = cv2.imread('p.jpg',cv2.IMREAD_COLOR)
