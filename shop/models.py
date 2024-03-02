from django.db import models
from django.utils import timezone
from datetime import datetime


class Study(models.Model):
    ticker = models.CharField(max_length=12)
    description = models.CharField(max_length=255)
    startDate = models.DateField
    endDate = models.DateField

class Category(models.Model):
    title = models.CharField(max_length=255)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):   # returns own title value in admin panel
        return self.title

class Course(models.Model):
    title = models.CharField(max_length=255)
    price = models.FloatField()
    students_qty = models.IntegerField()
    reviews_qty = models.IntegerField()
    category = models.ForeignKey(Category, on_delete=models.CASCADE)

    def __str__(self):   # returns own title value in admin panel
        return self.title + '. Price: ' + str(self.price)
    
    

class StockData(models.Model):
    # Django automatically creates an 'id' field if you don't specify one, so it's not needed here.
    ticker = models.CharField(max_length=10, db_index=True)  # Indexed by default
    volume = models.BigIntegerField()  # For large numbers
    vw = models.FloatField()  # Volume Weighted Average Price
    open = models.FloatField()  # Open price
    close = models.FloatField()  # Close price
    high = models.FloatField()  # High price
    low = models.FloatField()  # Low price
    timestamp = models.BigIntegerField(db_index=True)  # UNIX timestamp in milliseconds, indexed
    transactions = models.IntegerField()  # Number of transactions
    timeframe = models.CharField(max_length=6)  # Timeframe (e.g., "1Day", "1Hr")
    
    def __str__(self):
        # Correcting the __str__ method to reflect the fields available in this model
        return f"{self.ticker} @ {self.timestamp} - TF: {self.timeframe}"
