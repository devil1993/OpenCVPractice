import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==================================================================================================================STEP1============
# img = cv2.imread('p.jpg',cv2.IMREAD_GRAYSCALE)
# print(img)
# cv2.imshow('Image00', img)
# cv2.waitKey()

# cv2.destroyAllWindows()


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

# # ROI : Region of Image?? Region of interest??

# roi = img[100:250,100:250]
# # img[100:250,100:250] = [255,255,255]

# img[300:450,300:450] = roi
# cv2.imshow('Image00', img)
# cv2.waitKey(0)

# cv2.destroyAllWindows()

# =====================================================================================================================STEP5==========
img1 = cv2.imread('a.JPG')
img2 = cv2.imread('b.JPG')
img3 = cv2.imread('c.jpg')

# add = img1 + img2
# add = cv2.add(img1,img2)

# weighted = cv2.addWeighted(img1,0.6, img2, 0.4,0)

rows, cols, channels = img3.shape
roi = img1[0:rows,0:cols]

img2grey = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
rowset, mask = cv2.threshold(img2grey,150,255,cv2.THRESH_BINARY_INV)

# cv2.imshow('mask', mask)

mask_inv = cv2.bitwise_not(mask)

img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)
img3_fg = cv2.bitwise_and(img3, img3, mask = mask)

dst = cv2.add(img1_bg, img3_fg)

img1[0:rows,0:cols] = dst

cv2.imshow('img',img1)
# cv2.imshow('add', weighted)
cv2.waitKey(0)
cv2.destroyAllWindows()