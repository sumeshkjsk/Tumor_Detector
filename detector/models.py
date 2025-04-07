from django.db import models
from django.contrib.auth.models import User

class PredictionLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    image_name = models.CharField(max_length=255)
    prediction_result = models.CharField(max_length=50)
    confidence = models.FloatField()
    raw_prediction = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)

    class Meta:
        ordering = ['-timestamp']
        verbose_name = 'Prediction Log'
        verbose_name_plural = 'Prediction Logs'

    def __str__(self):
        return f"{self.image_name} - {self.prediction_result} ({self.confidence}%)"