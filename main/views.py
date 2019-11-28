

from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.utils import timezone

import pandas as pd
import numpy as np

from datetime import datetime 
from django.utils import timezone

from .models import Booking
from .modeling import *
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

def available(request):
	
	bookings = Booking.objects.all()
		
	data = modelConfig()

	for booking in bookings:
		if(booking.end_time > timezone.now()):
			data.loc[0]['available'][booking.parking_spot_id] = 0
	
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

		bookings = Booking.objects.all()
		
		data = modelConfig()

		for booking in bookings:
			if(booking.end_time > timezone.now()):
				data.loc[0]['available'][booking.parking_spot_id] = 0

		context = {
			'title'  : 'Booking',
			'availabe' : data.loc[0]['available'],
		}

		return render(request, 'booking.html', context)
		#return data.loc[0]['available'] 
		#return HttpResponse(data.loc[0]['available'])