from django.contrib import admin
from django.urls import path, include

from django.conf.urls.static import static
from django.conf import settings
urlpatterns = [
      path('admin/', admin.site.urls),
    # 使用 include() 将 users 应用的 urls 模块包含进来
    path('', include('imgup.urls'))
    
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
