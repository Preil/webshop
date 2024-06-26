from api.models import CategoryResource, CourseResource, IndicatorResource, StockDataResource, StudyResource, StudyIndicatorResource, NnModelResource, TrainedNnModelResource
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

api.register(CourseResource())
api.register(CategoryResource())
api.register(StockDataResource())
api.register(StudyResource())
api.register(IndicatorResource())
api.register(StudyIndicatorResource())
api.register(NnModelResource())
api.register(TrainedNnModelResource())


urlpatterns = [
    path('', include(api.urls))
]
