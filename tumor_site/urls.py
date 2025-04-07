# tumor_site/urls.py
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include  # ← don't forget this import

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('detector.urls')),  # ← this is correct
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)