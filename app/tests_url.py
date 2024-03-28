from django.test import TestCase
from django.urls import reverse ,resolve
from .views import index,map,detected


class TestUrls(TestCase):

    def test_render_index(self):
        url = reverse('index')
        self.assertEquals(resolve(url).func,index)

    def test_render_map(self):
        url = reverse('map')
        self.assertEquals(resolve(url).func,map)

    def test_render_detected(self):
        url = reverse('detected')
        self.assertEquals(resolve(url).func,detected)
