from app.models import Employee as Person
from app.models import Detected
from django.test import TestCase
from django.urls import reverse ,resolve
from .views import index,map,detected
from django.db import models
from datetime import datetime
import os

class Model_testing(TestCase):


    def test_check_count(self):
        l = Person.objects.all().count()
        self.P1= Person.objects.create(id='50', name ='dark')
        self.assertEquals(Person.objects.all().count(),l+1)

    # def test_check_count1(self):
    #     P= Person.objects.create(id='53', name ='lp')
    #     l = Detected.objects.all().count()
    #     t= models.DateTimeField()
    #     p = models.ImageField(upload_to='detected/', default='app/facerec/detected/noimg.png')
    #     Detected.objects.create(emp_id=P.id,time_stamp = t, photo = p)
    #     self.assertEquals(Detected.objects.all().count(),l+1)
