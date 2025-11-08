from django.db import models
from django.utils import timezone
from datetime import datetime

import base64
from tensorflow.keras.models import model_from_json


class Study(models.Model):
    ticker = models.CharField(max_length=12)
    timeFrame = models.CharField(max_length=12, default='1day')
    description = models.CharField(max_length=255)
    startDate = models.DateTimeField(default=timezone.now)
    endDate = models.DateTimeField(default=timezone.now)
    createdOn = models.DateTimeField(default=timezone.now)
    priceNormalizer = models.ForeignKey('StudyIndicator', on_delete=models.SET_NULL, related_name='priceNormalizer', null=True)
    volumeNormalizer = models.ForeignKey('StudyIndicator', on_delete=models.SET_NULL, related_name='volumeNormalizer', null=True)    
    

    def __str__(self):   # returns own ticker & description value in admin panel
        return self.ticker + ' ' + str(self.description)+' ' + str(self.id)
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
class Indicator(models.Model):
    name = models.CharField(max_length=40)
    description = models.CharField(max_length=255)
    functionName = models.CharField(max_length=50)
    parameters  = models.CharField(max_length=255)
    NORMALIZATION_TYPE_CHOICES = [
        ('NONE', 'Not normalized'),
        ('VOLUME', 'Volume normalized'),
        ('PRICE', 'Price normalized'),
    ]
    normalizationType = models.CharField(max_length=10, choices=NORMALIZATION_TYPE_CHOICES)

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
    
class NnModel(models.Model):
    ACTIVATION_FUNCTION_CHOICES = [
        ('relu', 'ReLU'),
        ('sigmoid', 'Sigmoid'),
        ('tanh', 'Tanh'),
    ]
    LOSS_FUNCTION_CHOICES = [
        ('mse', 'Mean Squared Error'),
        ('binary_crossentropy', 'Binary Crossentropy'),
        ('categorical_crossentropy', 'Categorical Crossentropy'),
    ]
    OPTIMIZER_CHOICES = [
        ('sgd', 'Stochastic Gradient Descent'),
        ('adam', 'Adam'),
        ('rmsprop', 'RMSprop'),
    ]

    name = models.CharField(max_length=40)
    description = models.CharField(max_length=255, blank=True)
    number_of_layers = models.IntegerField()
    nodes_per_layer = models.CharField(max_length=255)  # You can store this as a comma-separated string
    activation_function = models.CharField(max_length=50, choices=ACTIVATION_FUNCTION_CHOICES)
    loss_function = models.CharField(max_length=50, choices=LOSS_FUNCTION_CHOICES)
    optimizer = models.CharField(max_length=50, choices=OPTIMIZER_CHOICES)
    learning_rate = models.FloatField()
    batch_size = models.IntegerField()
    number_of_epochs = models.IntegerField()
    study = models.ForeignKey(Study, on_delete=models.CASCADE)

    def __str__(self):
        return self.name
    
class TrainedNnModel(models.Model):
    nn_model = models.ForeignKey(NnModel, on_delete=models.CASCADE)
    study = models.ForeignKey(Study, on_delete=models.CASCADE)
    serialized_model = models.BinaryField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Trained Model for {self.nn_model.name} at {self.created_at}"
    
    def save_model(self, model):
        serialized_model = model.to_json()
        encoded_model = base64.b64encode(serialized_model.encode('utf-8'))
        self.serialized_model = encoded_model
        self.save()

    def load_model(self):
        decoded_model = base64.b64decode(self.serialized_model)
        model_json = decoded_model.decode('utf-8')
        model = model_from_json(model_json)
        return model