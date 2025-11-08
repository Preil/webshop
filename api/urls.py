
from api.resources.TradingSessionResource import TradingSessionResource
from api.resources.StockDataResource import StockDataResource
from api.resources.StudyResource import StudyResource
from api.resources.IndicatorResource import IndicatorResource
from api.resources.StudyIndicatorResource import StudyIndicatorResource
from api.resources.StudyOrderResource import StudyOrderResource
from api.resources.TradingPlanResource import TradingPlanResource
from api.resources.StudyTradingPlanResource import StudyTradingPlanResource
from api.resources.NnModelResource import NnModelResource
from api.resources.TrainedNnModelResource import TrainedNnModelResource
from api.resources.SessionPotentialOrderResources import SessionPotentialOrderResource
from api.resources.SessionOrderResource import SessionOrderResource
from api.resources.SessionFillResource import SessionFillResource
from api.resources.SessionStockDataResource import SessionStockDataResource
from api.resources.SessionStockDataIndicatorValueResource import SessionStockDataIndicatorValueResource
from api.resources.SessionSettingsResource import SessionSettingsResource


from tastypie.api import Api
from django.urls import path, include

# /api/categories/      GET         all categories
# /api/courses/         GET, POST   all courses

# /api/categories/2/    GET         Single category
# /api/courses/3/       GET, DELETE Single cours

# For DELETE, POST requests enable Authorization header
# Example: ApiKey admin:admin123
# For using ApiKeys need to register Tastypie app insisde base/settings.py

api = Api(api_name='v1')

api.register(StockDataResource())
api.register(StudyResource())
api.register(IndicatorResource())
api.register(StudyIndicatorResource())
api.register(StudyOrderResource())
api.register(TradingPlanResource())
api.register(StudyTradingPlanResource())
api.register(NnModelResource())
api.register(TrainedNnModelResource())
api.register(TradingSessionResource())
api.register(SessionPotentialOrderResource())
api.register(SessionOrderResource())
api.register(SessionFillResource())
api.register(SessionStockDataResource())
api.register(SessionStockDataIndicatorValueResource())
api.register(SessionSettingsResource())

urlpatterns = [
    path('', include(api.urls))
]
