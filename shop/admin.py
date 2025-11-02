from django.contrib import admin

from . import models
from shop.session_models import (
    TradingSession, SessionPotentialOrder, 
    SessionOrder, SessionFill, SessionStockData, 
    SessionStockDataIndicatorValue, SessionSettings
)
admin.site.register(models.Category)
admin.site.register(models.Course)
admin.site.register(models.StockData)

admin.site.register(models.Study)
admin.site.register(models.Indicator)
admin.site.register(models.StudyIndicator)
admin.site.register(models.StudyStockDataIndicatorValue)

admin.site.register(models.StudyOrder)

admin.site.register(models.TradingPlan)
admin.site.register(models.StudyTradingPlan)
admin.site.register(models.NnModel)
admin.site.register(models.TrainedNnModel)
admin.site.register(TradingSession)
admin.site.register(SessionPotentialOrder)
admin.site.register(SessionOrder)
admin.site.register(SessionFill)
admin.site.register(SessionStockData)
admin.site.register(SessionStockDataIndicatorValue)
admin.site.register(SessionSettings)