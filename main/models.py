from django.db import models
from django.core.validators import MaxValueValidator
from datetime import datetime 
from django.utils import timezone

# Create your models here.
class Booking(models.Model):
	email = models.EmailField()
	parking_spot_id = models.PositiveIntegerField()
	start_time = models.DateTimeField(default = timezone.now)
	end_time = models.DateTimeField()
	pin = models.PositiveIntegerField()

	def __str__(self):
		return self.email