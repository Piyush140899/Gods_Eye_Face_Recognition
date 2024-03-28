
from django.test import TestCase,Client
from django.urls import reverse ,resolve
from .views import index,map,detected
from django.db import models
from datetime import datetime
import os


class TestViews(TestCase):

    def setUp(self):
        self.client =Client()

    def test_post(self):
        response = self.client.post(reverse('add_emp'))

        self.assertEquals(response.status_code,200)
        self.assertTemplateUsed(response , 'app/add_emp.html')
