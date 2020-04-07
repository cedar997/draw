
from django.urls import path
from . import views
from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
app_name = 'imgup'
urlpatterns = [
     
     path('',views.index,name="index"),
     path('up',views.upLoad,name="up"),
     path('admin/', admin.site.urls),
     path('show/',views.showImg,name="show")
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)