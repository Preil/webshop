from django.contrib import admin

from . import models

admin.site.register(models.Category)
admin.site.register(models.Course)
admin.site.register(models.StockData)
admin.site.register(models.Study)
admin.site.register(models.Indicator)
admin.site.register(models.StudyIndicator)
admin.site.register(models.StudyStockDataIndicatorValue)
admin.site.register(models.StudyOrder)
admin.site.register(models.TradingPlan)
