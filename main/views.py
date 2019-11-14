from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.utils import timezone

import matplotlib.pyplot as plt
from matplotlib.pyplot import imread,imshow
import pandas as pd
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.engine.topology import Layer, InputSpec
from .custom import LocalResponseNormalization
import cv2
from os.path import isfile, join, exists
import shutil
#import sys


from .models import Booking
from django.core.mail import send_mail
from django.conf import settings

from random import randint

# Create your views here.

#this is index html view
def index(request):
    context = {
        'title' : 'Home',
    }
    return render(request, 'index.html', context)


def direction(request):
	context = {
		'title' : 'Direction',
	}
	return render(request, 'direction.html', context)


data = pd.DataFrame(columns=['spot_id','latitude','longitute','rows','cols','slot_available','coordinates','available','price','name','contact'])

data.loc[0] = [
    1,
    28.5946,
    77.31994569999999,
    4,
    2,
    3,
    [
        [95,40,194,111],
        [95,140,195,198],
        [95,216,195,265],
        [93,309,195,380],
        [238,35,329,108],
        [231,131,334,201],
        [231,240,335,299],
        [228,309,338,380]
    ],
    [0, 0, 0, 0, 0, 0, 0, 0],
    1000,
    "Centralized Parking",
    "9999999999"
]

IMAGE_DIR = 'test_image/dropbox_image.jpeg'
CHECKPOINT_DIR = 'weights/checkpoint-07-0.07.hdf5'


def available(request):
	
	if not exists(IMAGE_DIR):
	    print('Image file missing... Exiting!!')
	    #sys.exit(0)

	if not exists(CHECKPOINT_DIR):
	    print('Checkpoint file missing... Exiting!!')
	    #sys.exit(0)

	model = load_model(CHECKPOINT_DIR, custom_objects={'LocalResponseNormalization': LocalResponseNormalization})


	f = plt.figure(figsize=(100,150))
	f.add_subplot(1,2, 1)
	im = imread(IMAGE_DIR)
	plt.imshow(im)
	im = Image.fromarray(im)
	im = im.resize((500, 400))
	im = np.array(im)

	images = []
	for cord in data['coordinates'][0]:
	  im_ = Image.fromarray(im[cord[1]:cord[3],cord[0]:cord[2]])
	  im_ = im_.resize((54, 32))
	  im_ = np.array(im_)
	  im_ = im_.transpose(1,0,2)
	  images.append(im_)

	images = np.array(images)

	predictions = model.predict(images, verbose=1)

	#print(predictions)
	
	result = []
	for p in range(len(predictions)):
		result.append(int(round(predictions[p][0])))

	print(result)

	predictions = np.hstack(predictions < 0.5).astype(int)
	#print("flag")
	i = 0
	im_ = np.copy(im)
	for cord in data['coordinates'][0]:
	  im_ = cv2.rectangle(im_,(cord[0],cord[1]),(cord[2],cord[3]),(255*int(predictions[i]),255*int(1-predictions[i]),0),2)
	  i += 1
	f.add_subplot(1,2, 2)
	#     print(im_)
	#    imshow(im_)
	#     plt.savefig('check.png')
	#imgplot = plt.imshow(im_)
	#plt.show()
	#data.loc[0]['available'] = result
	

	bookings = Booking.objects.all()

	for booking in bookings:
		if(booking.end_time > timezone.now()):
			result[booking.parking_spot_id] = 0


	for i in range(len(data.loc[0]['available'])):
		data.loc[0]['available'][i] = result[i]

	
	context = {
		'title'  : 'Available',
		'availabe' : data.loc[0]['available'],
	}

	return render(request, 'available.html', context)
	#return data.loc[0]['available'] 
	#return HttpResponse(data.loc[0]['available'])




def booking(request):
	
	if request.method == 'POST':
		email = request.POST.get('email')
		parking_spot_id = int(request.POST.get('parking_spot_id'))
		end_time = request.POST.get('end_time')


		pin = randint(1000, 9999)

		new_booking = Booking.objects.create(email = email, parking_spot_id = parking_spot_id-1, end_time= end_time, pin = pin)
		new_booking.save()

		subject = 'Pin for parking spot booking'
		message = ' Hello user, your pin for parking spot booking is {}. Your parking spot id is {} and your booking is valid till {}.'.format(pin, parking_spot_id, end_time)
		email_from = settings.EMAIL_HOST_USER
		recipient_list = [email,]
		send_mail( subject, message, email_from, recipient_list )
	    

		return redirect('booking')



	else:
		if not exists(IMAGE_DIR):
		    print('Image file missing... Exiting!!')
		    #sys.exit(0)

		if not exists(CHECKPOINT_DIR):
		    print('Checkpoint file missing... Exiting!!')
		    #sys.exit(0)

		model = load_model(CHECKPOINT_DIR, custom_objects={'LocalResponseNormalization': LocalResponseNormalization})



		f = plt.figure(figsize=(100,150))
		f.add_subplot(1,2, 1)
		im = imread(IMAGE_DIR)
		plt.imshow(im)
		im = Image.fromarray(im)
		im = im.resize((500, 400))
		im = np.array(im)

		images = []
		for cord in data['coordinates'][0]:
		  im_ = Image.fromarray(im[cord[1]:cord[3],cord[0]:cord[2]])
		  im_ = im_.resize((54, 32))
		  im_ = np.array(im_)
		  im_ = im_.transpose(1,0,2)
		  images.append(im_)

		images = np.array(images)

		predictions = model.predict(images, verbose=1)

		#print(predictions)
		
		result = []
		for p in range(len(predictions)):
			result.append(int(round(predictions[p][0])))

		print(result)

		predictions = np.hstack(predictions < 0.5).astype(int)
		#print("flag")
		i = 0
		im_ = np.copy(im)
		for cord in data['coordinates'][0]:
		  im_ = cv2.rectangle(im_,(cord[0],cord[1]),(cord[2],cord[3]),(255*int(predictions[i]),255*int(1-predictions[i]),0),2)
		  i += 1
		f.add_subplot(1,2, 2)
		#     print(im_)
		#    imshow(im_)
		#     plt.savefig('check.png')
		#imgplot = plt.imshow(im_)
		#plt.show()
		#data.loc[0]['available'] = result
		

		bookings = Booking.objects.all()

		for booking in bookings:
			if(booking.end_time > timezone.now()):
				result[booking.parking_spot_id] = 0


		for i in range(len(data.loc[0]['available'])):
			data.loc[0]['available'][i] = result[i]

		context = {
			'title'  : 'Booking',
			'availabe' : data.loc[0]['available'],
		}

		return render(request, 'booking.html', context)
		#return data.loc[0]['available'] 
		#return HttpResponse(data.loc[0]['available'])