from django.db import models

# Create your models here.

class img(models.Model):
    image = models.ImageField(upload_to='colors/img/')
    
