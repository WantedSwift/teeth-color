from django.shortcuts import render
import os
# Create your views here.
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
from django.forms.models import model_to_dict
from pathlib import Path
import operator
import json

from scipy.linalg import norm
from scipy import sum, average
from colors.models import img

import numpy as np
import argparse
import cv2 as cv 
from cv2 import bitwise_and as output0


@csrf_exempt
def upload_image_user(request):
  #  try:
        image = request.FILES.get('image')
        
        models1 = img.objects.create(image=image)
        image = cv.imread(str(models1.image))
		
        list_response = start_pars2(image)
       
  
        return JsonResponse({
            'status': True,
            'data': list_response,
        }, encoder=json.JSONEncoder)

    #except Exception as exc:
   # except Exception as exc:
    #    return JsonResponse({
      #      'status': False,
      #      'message': "invalid user"
      #  })


def start_pars2(image):
	imgn = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	imgn = cv.resize(imgn, (47, 73)) 


	images = Path("colors/teeh_image").glob("*.png")
	image_strings = [str(p) for p in images]
	print('hi',image_strings)
	list2 = {}

	for item in image_strings:
		img3 = cv.imread(item)
		img3 = cv.resize(img3, (47, 73))
		item2 = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)
		error , diff = mse(item2,imgn)
		item = item.replace("colors/teeh_image/","")
		item = item.replace(".png","")
		list2[item] = error
		print(item,error)

	list2 = sorted(list2.items(), key=operator.itemgetter(1))
	list2 = list2[0],list2[1],list2[2]
	list2 = list2

	return list2


		



def mse(img1, img2):
   h, w = img1.shape
   diff = cv.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse, diff

def to_grayscale(arr):
    "If arr is a color image (3D array), convert it to grayscale (2D array)."
    if len(arr.shape) == 3:
        return average(arr, -1)  # average over the last axis (color channels)
    else:
        return arr

def compare_images(img1):
    # normalize to compensate for exposure difference, this may be unnecessary
    # consider disabling it
    img1 = normalize(img1)
   
    # calculate the difference and its norms
   
    m_norm = sum(abs(img1))  # Manhattan norm
    z_norm = norm(img1.ravel(), 0)  # Zero norm
    return (z_norm)


def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng



# construct the argument parse and parse the arguments


def start_pars(image):

#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", help = "path to the image")
#args = vars(ap.parse_args())


# define the list of boundaries
#NEWA1 = [160, 195, 205], [238, 237, 240]
	A1 = [158, 199, 209], [207, 225, 236]
	A2 = [124, 191, 205], [199, 224, 236]
	A3 = [126, 187, 203], [196, 220, 237]
	A35 = [112, 178, 200], [200, 217, 235]
	A4 = [94, 166, 197], [194, 208, 226]

	B1 = [168, 200, 206], [204, 229, 237]
	B2 = [143, 193, 205], [200, 225, 237]
	B3 = [112, 185, 203], [200, 222, 237]
	B4 = [82, 172, 203], [191, 213, 233]

	C1 = [139, 188, 200], [203, 220, 232]
	C2 = [117, 178, 198], [200, 217, 232]
	C3 = [109, 174, 195], [195, 211, 227]
	C4 = [79, 154, 188], [218, 225, 230]

	D2 = [135, 188, 202], [203, 218, 231]
	D3 = [131, 183, 201], [202, 218, 232]
	D4 = [115, 181, 201], [206, 220, 230]


	boundaries = [
	(A1),
	(A2),
	(A3),
	(A35),
	(A4),
	(B1),
	(B2),
	(B3),
	(B4),
	(C1),
	(C2),
	(C3),
	(C4),
	(D2),
	(D3),
	(D4)
	]

	outputA1 = output0
	outputA2 = output0
	outputA3 = output0
	outputA35 = output0
	outputA4 = output0

	outputB1 = output0
	outputB2 = output0
	outputB3 = output0
	outputB4 = output0

	outputC1 = output0
	outputC2 = output0
	outputC3 = output0
	outputC4 = output0

	outputD2 = output0
	outputD3 = output0
	outputD4 = output0
# loop over the boundaries
	i = 0
	for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")
		print("i am i",i)
	# find the colors within the specified boundaries and apply
	# the mask
		mask = cv.inRange(image, lower, upper)
		if i == 0 :
		
			outputA1 = cv.bitwise_and(image, image, mask = mask)
		elif i == 1:
			outputA2 = cv.bitwise_and(image, image, mask = mask)
		elif i == 2:
			outputA3 = cv.bitwise_and(image, image, mask = mask)
		elif i == 3:
			outputA35 = cv.bitwise_and(image, image, mask = mask)
		elif i == 4:
			outputA4 = cv.bitwise_and(image, image, mask = mask)
		elif i == 5:
			outputB1 = cv.bitwise_and(image, image, mask = mask)
		elif i == 6:
			outputB2 = cv.bitwise_and(image, image, mask = mask)
		elif i == 7:
			outputB3 = cv.bitwise_and(image, image, mask = mask)
		elif i == 8:
			outputB4 = cv.bitwise_and(image, image, mask = mask)
		elif i == 9:
			outputC1 = cv.bitwise_and(image, image, mask = mask)
		elif i == 10:
			outputC2 = cv.bitwise_and(image, image, mask = mask)
		elif i == 11:
			outputC3 = cv.bitwise_and(image, image, mask = mask)
		elif i == 12:
			outputC4 = cv.bitwise_and(image, image, mask = mask)
		elif i == 13:
			outputD2 = cv.bitwise_and(image, image, mask = mask)
		elif i == 14:
			outputD3 = cv.bitwise_and(image, image, mask = mask)
		else:
			outputD4 = cv.bitwise_and(image, image, mask = mask)



		i = i + 1



#print("output1",np.asarray(output1, dtype=np.uint8))
#print("output2",np.asarray(output2, dtype=np.uint8))

	imgA1 = to_grayscale(outputA1.astype(float))
	imgA2 = to_grayscale(outputA2.astype(float))
	imgA3 = to_grayscale(outputA3.astype(float))
	imgA35 = to_grayscale(outputA35.astype(float))
	imgA4 = to_grayscale(outputA4.astype(float))

	imgB1 = to_grayscale(outputB1.astype(float))
	imgB2 = to_grayscale(outputB2.astype(float))
	imgB3 = to_grayscale(outputB3.astype(float))
	imgB4 = to_grayscale(outputB4.astype(float))

	imgC1 = to_grayscale(outputC1.astype(float))
	imgC2 = to_grayscale(outputC2.astype(float))
	imgC3 = to_grayscale(outputC3.astype(float))
	imgC4 = to_grayscale(outputC4.astype(float))


	imgD2 = to_grayscale(outputD2.astype(float))
	imgD3 = to_grayscale(outputD3.astype(float))
	imgD4 = to_grayscale(outputD4.astype(float))
# compare 
	a_1 = compare_images(imgA1)
	a_2 = compare_images(imgA2)
	a_3 = compare_images(imgA3)
	a_4 = compare_images(imgA4)
	a_35 = compare_images(imgA35)

	b_1 = compare_images(imgB1)
	b_2 = compare_images(imgB2)
	b_3 = compare_images(imgB3)
	b_4 = compare_images(imgB4)

	c_1 = compare_images(imgC1)
	c_2 = compare_images(imgC2)
	c_3 = compare_images(imgC3)
	c_4 = compare_images(imgC4)

	d_2 = compare_images(imgD2)
	d_3 = compare_images(imgD3)
	d_4 = compare_images(imgD4)


	all = [a_1,a_2,a_3,a_35,a_4,
		b_1,b_2,b_3,b_4,
		c_1,c_2,c_3,c_4,
		d_2,d_3,d_4,
	]
	all2 = ["a_1","a_2","a_3","a_35","a_4",
		"b_1","b_2","b_3","b_4",
		"c_1","c_2","c_3","c_4",
		"d_2","d_3","d_4",
	]
	all3 = [a_1,a_2,a_3,a_35,a_4,
		b_1,b_2,b_3,b_4,
		c_1,c_2,c_3,c_4,
		d_2,d_3,d_4,
	]
	maxed = 0

	count = 0
	index_item = 0
	for item in all:

		if maxed <= item:
			maxed = item
			index_item = count

		count = count + 1

	first_lvl = all2[index_item]
	all3.pop(index_item)
	all2.pop(index_item)

	maxed2 = 0
	count2 = 0
	index_item2 = 0

	for item in all3:

		if maxed2 <= item:
			maxed2 = item
			index_item2 = count2

		count2 = count2 + 1

	first_lvl2 = all2[index_item2]
	all3.pop(index_item2)
	all2.pop(index_item2)

	maxed3 = 0
	count3 = 0
	index_item3 = 0

	for item in all3:

		if maxed3 <= item:
			maxed3 = item
			index_item3 = count3

		count3 = count3 + 1

	first_lvl3 = all2[index_item3]
	all3.pop(index_item3)
	all2.pop(index_item3)

	print('finaled =>',first_lvl,first_lvl2,first_lvl3)
	list11 = [first_lvl,first_lvl2,first_lvl3]
	return list11
		

"""
list = sorted(all)[::-1]
print("i am ok=>",sorted(all)[::-1])

print ("Manhattan norm:", a_1, "/ per pixel:", a_1/imgA1.size*100.0)
print ("Manhattan norm:", a_2, "/ per pixel:", a_2/imgA2.size*100.0)
print ("Manhattan norm:", a_3, "/ per pixel:", a_3/imgA3.size*100.0)
print ("Manhattan norm:", a_35, "/ per pixel:", a_35/imgA35.size*100.0)
print ("Manhattan norm:", a_4, "/ per pixel:", a_4/imgA4.size*100.0)

print ("Manhattan norm:", b_1, "/ per pixel:", b_1/imgB1.size*100.0)
print ("Manhattan norm:", b_2, "/ per pixel:", b_2/imgB2.size*100.0)
print ("Manhattan norm:", b_3, "/ per pixel:", b_3/imgB3.size*100.0)
print ("Manhattan norm:", b_4, "/ per pixel:", b_4/imgB4.size*100.0)

print ("Manhattan norm:", c_1, "/ per pixel:", c_1/imgC1.size*100.0)
print ("Manhattan norm:", c_2, "/ per pixel:", c_2/imgC2.size*100.0)
print ("Manhattan norm:", c_3, "/ per pixel:", c_3/imgC3.size*100.0)
print ("Manhattan norm:", c_4, "/ per pixel:", c_4/imgC4.size*100.0)

print ("Manhattan norm:", d_2, "/ per pixel:", d_2/imgD2.size*100.0)
print ("Manhattan norm:", d_3, "/ per pixel:", d_3/imgD3.size*100.0)
print ("Manhattan norm:", d_4, "/ per pixel:", d_4/imgD4.size*100.0)


print ("Zero norm:", list)

"""

#if a_1 > a_2 :
#	cv.imshow("images-Out1", np.hstack([image, output1, output2,output3,output4]))
#	cv.waitKey(0)	
##	cv.imshow("images-Out2", np.hstack([image, output1, output2,output3,output4]))
#	cv.waitKey(0)


#print ("Zero norm:", n_0, "/ per pixel:", n_0*1.0/img1.size)
#print ("Zero norm:", n_1, "/ per pixel:", n_0*1.0/img2.size)

#print("he he he he22",output2.size)

