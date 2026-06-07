from django.db import models


class FeedbackSample(models.Model):
    image         = models.ImageField(upload_to='feedback/')
    predicted     = models.CharField(max_length=10)   # what model said
    true_label    = models.CharField(max_length=10)   # user correction
    was_correct   = models.BooleanField()
    used_in_train = models.BooleanField(default=False)
    created_at    = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
