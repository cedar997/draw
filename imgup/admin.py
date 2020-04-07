from django.contrib import admin

# Register your models here.
from imgup.models import TagImage,Img
admin.site.register(TagImage)
admin.site.register(Img)