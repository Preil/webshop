from django.contrib import admin
from django import forms

from . import models
from .models import Study, StudyIndicator, StudyTradingPlan
from shop.session_models import (
    TradingSession, SessionPotentialOrder, 
    SessionOrder, SessionFill, SessionStockData, 
    SessionStockDataIndicatorValue, SessionSettings
)
class StudyAdminForm(forms.ModelForm):
    class Meta:
        model = Study
        fields = '__all__'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        inst = self.instance if getattr(self, 'instance', None) else None
        qs = StudyIndicator.objects.none()
        if inst and inst.pk:
            qs = StudyIndicator.objects.filter(study=inst)

        # Limit to indicators of this Study only
        for fld in ('priceNormalizer', 'volumeNormalizer', 'mainSatrIndicator'):
            if fld in self.fields:
                self.fields[fld].queryset = qs
                self.fields[fld].help_text = 'Pick an indicator created for this Study (e.g., sATR14).'

@admin.register(Study)
class StudyAdmin(admin.ModelAdmin):
    form = StudyAdminForm
    list_display = ('id', 'ticker', 'timeFrame', 'description')
    search_fields = ('ticker', 'description')

# admin.site.register(models.Study)

class TradingSessionAdminForm(forms.ModelForm):
    class Meta:
        model = TradingSession
        fields = '__all__'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        inst = self.instance if getattr(self, 'instance', None) else None
        qs = StudyTradingPlan.objects.none()
        if inst and inst.pk and getattr(inst, 'study_id', None):
            qs = StudyTradingPlan.objects.filter(study_id=inst.study_id)
        if 'studyTradingPlan' in self.fields:
            self.fields['studyTradingPlan'].queryset = qs
            self.fields['studyTradingPlan'].help_text = "Pick a plan attached to this session's Study."

@admin.register(TradingSession)
class TradingSessionAdmin(admin.ModelAdmin):
    form = TradingSessionAdminForm
    list_display = ('id', 'study', 'get_study_timeframe', 'studyTradingPlan', 'planLockedAt')
    search_fields = ('id', 'name', 'session_id')

    def get_study_timeframe(self, obj):
        return getattr(obj.study, 'timeFrame', None)
    get_study_timeframe.short_description = 'Timeframe'
    get_study_timeframe.admin_order_field = 'study__timeFrame'

@admin.register(SessionPotentialOrder)
class SessionPotentialOrderAdmin(admin.ModelAdmin):
    list_display = ("id", "session", "direction", "decision", "createdAt")
    list_filter = ("session", "direction", "decision")
    search_fields = ("id", "session__id", "direction")
    ordering = ("-id",)  # sort by id descending


admin.site.register(models.StockData)

admin.site.register(models.Indicator)
admin.site.register(models.StudyIndicator)
admin.site.register(models.StudyStockDataIndicatorValue)

admin.site.register(models.StudyOrder)

admin.site.register(models.TradingPlan)
admin.site.register(models.StudyTradingPlan)
admin.site.register(models.NnModel)
admin.site.register(models.TrainedNnModel)
# admin.site.register(TradingSession)
# admin.site.register(SessionPotentialOrder)
admin.site.register(SessionOrder)
admin.site.register(SessionFill)
admin.site.register(SessionStockData)
admin.site.register(SessionStockDataIndicatorValue)
admin.site.register(SessionSettings)