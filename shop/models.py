from django.db import models
from django.utils import timezone
from datetime import datetime


class Study(models.Model):
    ticker = models.CharField(max_length=12)
    timeFrame = models.CharField(max_length=12, default='1day')
    description = models.CharField(max_length=255)
    startDate = models.DateTimeField(default=timezone.now)
    endDate = models.DateTimeField(default=timezone.now)
    createdOn = models.DateTimeField(default=timezone.now)

    def __str__(self):   # returns own ticker & description value in admin panel
        return self.ticker + ' ' + str(self.description)+' ' + str(self.id)

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
    study = models.ForeignKey(Study, on_delete=models.CASCADE)  # Foreign key to the Study model
    
    def __str__(self):
        # Correcting the __str__ method to reflect the fields available in this model
        return f"{self.ticker} @ {self.timestamp} - TF: {self.timeframe}"

class StockDataNormal(models.Model):
    stockData = models.ForeignKey(StockData, on_delete=models.CASCADE)
    study = models.ForeignKey(Study, on_delete=models.CASCADE)
    normVolume = models.FloatField()  # Normalized Volume
    normVw = models.FloatField()  # Normlized Volume Weighted Average Price
    normOpen = models.FloatField()  # Normlized Open price
    normClose = models.FloatField()  # Normlized Close price
    normHigh = models.FloatField()  # Normlized High price
    normLow = models.FloatField()  # Normlized Low price    
    def __str__(self):
        # Correcting the __str__ method to reflect the fields available in this model
        return f"{self.stockData.ticker} @ {self.stockData.timestamp} - TF: {self.stockData.timeframe}"


class Indicator(models.Model):
    name = models.CharField(max_length=40)
    description = models.CharField(max_length=255)
    functionName = models.CharField(max_length=50)
    parameters  = models.CharField(max_length=255)

    def __str__(self):   # returns own ticker & description value in admin panel
        return self.name + ' ' + str(self.parameters)

# This model stores parameters for particular indicator that is used in a particular study    
class StudyIndicator(models.Model):
    study = models.ForeignKey(Study, on_delete=models.CASCADE)
    indicator = models.ForeignKey(Indicator, on_delete=models.CASCADE)
    mask = models.CharField(max_length=15)
    parametersValue = models.CharField(max_length=255)

    def __str__(self):   # returns own ticker & description value in admin panel
        return str(self.study) + ' ' + str(self.indicator) + ' ' + str(self.parametersValue)
    
# This model stores the values of the indicators for each stock data item of particular study
class StudyStockDataIndicatorValue(models.Model):
    stockDataItem = models.ForeignKey(StockData, on_delete=models.CASCADE)
    studyIndicator = models.ForeignKey(StudyIndicator, on_delete=models.CASCADE)
    value = models.CharField(max_length=255)

    def __str__(self):   # returns own ticker & description value in admin panel
        return str(self.studyIndicator.study) + ' ' + str(self.stockDataItem.pk) + ' ' + str(self.value)


# This model stores normalized values of the indicators StudyStockDataIndicatorValue
class StudyStockDataIndicatorNormalValue(models.Model):
    stockDataItem = models.ForeignKey(StockData, on_delete=models.CASCADE)
    studyIndicator = models.ForeignKey(StudyIndicator, on_delete=models.CASCADE)
    normalValue = models.CharField(max_length=255)

    def __str__(self):   # returns own ticker & description value in admin panel
        return str(self.studyIndicator.study) + ' ' + str(self.stockDataItem.pk) + ' ' + str(self.normalValue)
    
class StudyOrder(models.Model):
    study = models.ForeignKey(Study, on_delete=models.CASCADE)
    stockDataItem = models.ForeignKey(StockData, on_delete=models.CASCADE)
    ORDER_TYPE_CHOICES = [
        ('MARKET', 'Market'),
        ('LIMIT', 'Limit'),
    ]
    orderType = models.CharField(max_length=10, choices=ORDER_TYPE_CHOICES)
    quantity = models.IntegerField()
    limitPrice = models.DecimalField(max_digits=16, decimal_places=8, null=True, blank=True)
    takeProfit = models.DecimalField(max_digits=16, decimal_places=8, null=True, blank=True)
    stopLoss = models.DecimalField(max_digits=16, decimal_places=8, null=True, blank=True)
    DIRECTION_CHOICES = [
        ('BUY', 'Buy'),
        ('SELL', 'Sell'),
    ]
    direction = models.CharField(max_length=10, choices=DIRECTION_CHOICES)
    TIME_IN_FORCE_CHOICES = [
        ('GTC', 'Good Till Cancelled'),
        ('FOK', 'Fill Or Kill'),
        ('IOC', 'Immediate Or Cancel'),
    ]
    timeInForce = models.CharField(max_length=10, choices=TIME_IN_FORCE_CHOICES)
    STATUS_CHOICES = [
        ('OPEN', 'Open'),
        ('FILLED', 'Filled'),
        ('CANCELLED', 'Cancelled'),
        ('EXPIRED', 'Expired'),
        ('CLOSED_BY_SL', 'ClosedBySl'),
        ('CLOSED_BY_TP', 'ClosedByTp'),
    ]
    status = models.CharField(max_length=12, choices=STATUS_CHOICES)
    createdAt = models.DateTimeField(null=True, blank=True)
    filledAt = models.DateTimeField(null=True, blank=True)
    expiredAt = models.DateTimeField(null=True, blank=True)
    closedAt = models.DateTimeField(null=True, blank=True)
    cancelledAt = models.DateTimeField(null=True, blank=True)
    
    #Trading plan parameters 
    lpoffsetTP = models.DecimalField(max_digits=16, decimal_places=8, null=True, blank=True)
    slTP = models.DecimalField(max_digits=16, decimal_places=8, null=True, blank=True)
    tpTP = models.DecimalField(max_digits=16, decimal_places=8, null=True, blank=True)
    
class StudyOrderNormalValues(models.Model):
    studyOrder = models.ForeignKey(StudyOrder, on_delete=models.CASCADE)
    normLimitPrice = models.DecimalField(max_digits=16, decimal_places=8, null=True, blank=True)
    normTakeProfit = models.DecimalField(max_digits=16, decimal_places=8, null=True, blank=True)
    normStopLoss = models.DecimalField(max_digits=16, decimal_places=8, null=True, blank=True)
    
    def __str__(self):   # returns own ticker & description value in admin panel
        return str(self.studyOrder) + ' ' + str(self.normalValue)
class TradingPlan(models.Model):
    name = models.CharField(max_length=255)
    tradingPlanParams = models.CharField(max_length=255, default="{LPoffset: [0.1, 0.2, 0.3], stopLoss: [0.1, 0.2, 0.3], takeProfit: [3, 4, 5]}")

    def __str__(self):   # returns own ticker & description value in admin panel

        return str(self.name) + ' ' + str(self.tradingPlanParams)

class StudyTradingPlan(models.Model):
    study = models.ForeignKey(Study, on_delete=models.CASCADE)
    tradingPlan = models.ForeignKey(TradingPlan, on_delete=models.CASCADE)
    
    def __str__(self):   # returns own ticker & description value in admin panel
        return str(self.study) + ' ' + str(self.tradingPlan)