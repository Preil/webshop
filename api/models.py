from django.shortcuts import get_object_or_404
from django.core import serializers
from django.db import transaction
from tastypie.resources import ModelResource
from tastypie.utils import trailing_slash
from shop.models import (
    Category,
    Course,
    StockData,
    Study,
    Indicator,
    StudyIndicator,
    StudyStockDataIndicatorValue,
    StudyOrder,
    TradingPlan,
    StudyTradingPlan,
    NnModel,
    TrainedNnModel,
)
from shop.session_models import (
    TradingSession,
    SessionPotentialOrder,
    SessionOrder,
    SessionFill,
    SessionStockData,
)
from api.authentication import CustomApiKeyAuthentication
from tastypie.authorization import Authorization
from django.http import HttpResponse, JsonResponse
import json
from django.urls import re_path
import logging
from shop.services import calculateStudy, simulate_trades
import pandas as pd
import numpy as np
from pandas import json_normalize
from collections import OrderedDict
from decimal import Decimal
from shop.bakery import train_nn_model, check_training_status
logger = logging.getLogger(__name__)
from tastypie.exceptions import ImmediateHttpResponse
from django.http import HttpResponseBadRequest




# endpoints examples

# /api/categories/
# /api/courses/

# /api/categories/2/
# /api/courses/3/

class StockDataResource(ModelResource):
    class Meta:
        queryset = StockData.objects.all()
        resource_name = 'StockData'
        allowed_methods = ['get', 'delete', 'post']
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()

    def prepend_urls(self):
        return [
            re_path(r'^StockData/save_bulk/$', self.wrap_view('save_bulk'), name='api_save_bulk'),
        ]

    def save_bulk(self, request, *args, **kwargs):
        logger.info("save_bulk method is called")

        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        # Assuming POST request with JSON body
        data = json.loads(request.body)
        results = data.get('results', [])
        study_id = data.get('studyId')

        # Get the study object
        study = get_object_or_404(Study, id=study_id)

        for stock_data in results:
            StockData.objects.get_or_create(
                ticker=data.get('ticker'),
                timestamp=stock_data.get('t'),
                study=study,
                defaults={
                    'volume': stock_data.get('v'),
                    'vw': stock_data.get('vw'),
                    'open': stock_data.get('o'),
                    'close': stock_data.get('c'),
                    'high': stock_data.get('h'),
                    'low': stock_data.get('l'),
                    'transactions': stock_data.get('n'),
                    'timeframe': data.get('timeFrame'),
                }
            )
        self.log_throttled_access(request)
        return self.create_response(request, {'success': True, 'message': 'Stock data saved successfully'})
class CategoryResource(ModelResource):
    class Meta:
        queryset = Category.objects.all()
        resource_name = 'categories'
        allowed_methods = ['get']
class StudyResource(ModelResource):
    class Meta:
        queryset = Study.objects.all()
        resource_name = 'studies'
        allowed_methods = ['get', 'delete', 'post']
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()
    def prepend_urls(self):
        return [
            # /api/studies/1/stockdata/ - to get stock data for study with id 1
            re_path(r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/stockdata%s$' % (self._meta.resource_name, trailing_slash()), self.wrap_view('get_stockdata'), name="api_get_stockdata"),
            # /api/studies/1/indicators/ - to get indicators for study with id 1
            re_path(r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/indicators%s$' % (self._meta.resource_name, trailing_slash()), self.wrap_view('get_study_indicators'), name="api_get_study_indicators"),
            # /api/studies/1/indicators/ - to get trading plans for study with id 1
            re_path(r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/tradingPlans%s$' % (self._meta.resource_name, trailing_slash()), self.wrap_view('get_study_tradingplans'), name="api_get_study_tradingplans"),
            # /api/studies/1/indicatorsValues/ - to get indicators values for study with id 1
            re_path(r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/indicatorsValues%s$' % (self._meta.resource_name, trailing_slash()), self.wrap_view('get_study_indicators_values'), name="api_get_study_indicators_values"),
            # /api/studies/1/calculate/ - to calculate indicators values for study with id 1
            re_path(r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/calculate%s$' % (self._meta.resource_name, trailing_slash()), self.wrap_view('calculate'), name="api_calculate"),
            # /api/studies/1/tradingPlans/1/generate/ - to generat trades for study with id 1 according TradingPlan with id 1 (?P<tradingPlan_id>\d+)/
            re_path(r'^(?P<resource_name>%s)/(?P<pk>\d+)/tradingPlans/(?P<tradingPlan_id>\d+)/generate%s$' % (self._meta.resource_name, trailing_slash()), self.wrap_view('generate_trades'), name="api_generate"),
            # /api/studies/1/orders/ - to get orders for study with id 1
            re_path(r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/orders%s$' % (self._meta.resource_name, trailing_slash()), self.wrap_view('get_study_orders'), name="api_get_study_orders"),
            # /api/studies/1/summaryData/ - to get summary data for study with id 1
            re_path(r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/summaryData%s$' % (self._meta.resource_name, trailing_slash()), self.wrap_view('get_summary_data'), name="api_get_summary_data"),
            # /api/studies/1/normalizedData/ - to get normalized data for study with id 1
            re_path(r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/normalizedData%s$' % (self._meta.resource_name, trailing_slash()), self.wrap_view('get_normalized_data'), name="api_get_normalized_data"),
            # /api/studies/1/models/ - to get models for study with id 1
            re_path(r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/models%s$' % (self._meta.resource_name, trailing_slash()), self.wrap_view('get_study_models'), name="api_get_study_models"),
            # /api/studies/1/models/1/train/ - to train model with id 1 for study with id 1
            re_path(r'^(?P<resource_name>%s)/(?P<pk>\d+)/models/(?P<model_id>\d+)/train%s$' % (self._meta.resource_name, trailing_slash()), self.wrap_view('train_model'), name="api_train_model"),
            # /api/studies/1/models/1/status/ - to check training status of model with id 1 for study with id 1
            re_path(r'^(?P<resource_name>%s)/models/(?P<model_id>\d+)/status%s$' % (self._meta.resource_name, trailing_slash()), self.wrap_view('check_status'), name="api_check_status"),
            # /api/v1/studies/1/trainedModels/ - to get trained models for study with id 1
            re_path(r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/trainedModels%s$' % (self._meta.resource_name, trailing_slash()), self.wrap_view('get_trained_models'), name="api_get_trained_models"),
            # /api/v1/studies/1/trainedModels/1/ - to get trained model with id 1 for study with id 1
            re_path(r'^(?P<resource_name>%s)/(?P<studyId>\d+)/trainedModels/(?P<trainedModelId>\d+)%s$' % (self._meta.resource_name, trailing_slash()), self.wrap_view('get_trained_model_detail'), name="api_get_trained_model_detail"),
        ]
    


    def get_study_orders(self, request, **kwargs):
        try:
            study = Study.objects.get(pk=kwargs['pk'])
        except Study.DoesNotExist:
            return self.create_response(request, {'error': 'not found'}, Http404)

        orders = StudyOrder.objects.filter(study=study).order_by('createdAt').values()
        orders_list = list(orders)  # Convert the QuerySet to a list
        return self.create_response(request, orders_list)

    def get_study_indicators(self, request, **kwargs):
        try:
            study = Study.objects.get(pk=kwargs['pk'])
        except Study.DoesNotExist:
            return self.create_response(request, {'error': 'not found'}, Http404)

        indicators = StudyIndicator.objects.filter(study=study)
        data = []
        for indicator in indicators:
            data.append({
                'id': indicator.id,
                'mask': indicator.mask,
                'indicator_id': indicator.indicator.id,
                'indicator_name': indicator.indicator.name,
                'indicator_function': indicator.indicator.functionName,
                'indicator_parameters': indicator.indicator.parameters,
                'parametersValue': indicator.parametersValue,
            })

        return JsonResponse(data, safe=False)
    
    def get_study_indicators_values(self, request, **kwargs):
        try:
            study = Study.objects.get(pk=kwargs['pk'])
        except Study.DoesNotExist:
            return self.create_response(request, {'error': 'not found'}, Http404)

        indicators_values = StudyStockDataIndicatorValue.objects.filter(stockDataItem__study=study)
        indicators_values_list = []
        for indicator_value in indicators_values:
            indicators_values_list.append({
                'id': indicator_value.id,
                'stockDataItem_id': indicator_value.stockDataItem_id,
                'stockDataItem_timestamp': indicator_value.stockDataItem.timestamp,
                'studyIndicator_id': indicator_value.studyIndicator_id,
                'value': indicator_value.value,
                'indicator_name': indicator_value.studyIndicator.indicator.name,
                'indicator_mask': indicator_value.studyIndicator.mask,
            })

        return self.create_response(request, indicators_values_list)
    
    def get_study_tradingplans(self, request, **kwargs):
        try:
            study = Study.objects.get(pk=kwargs['pk'])
        except Study.DoesNotExist:
            return self.create_response(request, {'error': 'not found'}, Http404)

        tradingplans = StudyTradingPlan.objects.filter(study=study)
        data = []
        for tradingplan in tradingplans:
            data.append({
                'id': tradingplan.id,
                'tradingplan_id': tradingplan.tradingPlan.id,
                'tradingplan_name': tradingplan.tradingPlan.name,
                'tradingplan_parameters': tradingplan.tradingPlan.tradingPlanParams,
            })

        return JsonResponse(data, safe=False)
    
    def get_stockdata(self, request, **kwargs):
        try:
            study = Study.objects.get(pk=kwargs['pk'])
        except Study.DoesNotExist:
            return self.create_response(request, {'error': 'not found'}, Http404)

        stockdata = StockData.objects.filter(study=study)
        stockdata_json = serializers.serialize('json', stockdata)

        return self.create_response(request, stockdata_json)
    
    def calculate(self, request, **kwargs):
        # Basic method to check HTTP method and call the actual calculation method
        self.method_check(request, allowed=['get'])
        self.is_authenticated(request)
        self.throttle_check(request)

        # Get the study object
        study = Study.objects.get(pk=kwargs['pk'])

        # Perform the calculations
        result = calculateStudy(study)

        self.log_throttled_access(request)
        return self.create_response(request, {'result': result})

    def generate_trades(self, request, **kwargs):
        # Basic method to check HTTP method and call the actual calculation method
        self.method_check(request, allowed=['get'])
        self.is_authenticated(request)
        self.throttle_check(request)

        print("Starting generate trades for study id: ", kwargs)
        # Get the study object
        study = Study.objects.get(pk=kwargs['pk'])
        tradingPlan = TradingPlan.objects.get(pk=kwargs['tradingPlan_id'])  # Change this line
        print("Starting generate trades for study id: ", study.id)
        print("Trading plan id: ", tradingPlan.id)
        # Perform trades simulations
        result = simulate_trades(study, tradingPlan.tradingPlanParams)

        self.log_throttled_access(request)
        return self.create_response(request, {'result': result})  

    # Get normalized data for a study 
    def get_normalized_data(self, request, **kwargs):
        try:
            study = Study.objects.get(pk=kwargs['pk'])
        except Study.DoesNotExist:
            return self.create_response(request, {'error': 'not found'}, Http404)

        # Function to get the normalization type of an indicator using 'mask'
        def get_indicator_normalization_type(indicator_mask):
            indicators = StudyIndicator.objects.filter(mask=indicator_mask)
            if indicators.count() == 1:
                return indicators.first().indicator.normalizationType  # Follow the reference to the Indicator object
            else:
                raise ValueError(f"Multiple or no StudyIndicator objects found for mask '{indicator_mask}'")

        # Fetch the 'mask' value of the priceNormalizer studyIndicator and append "value"
        price_normalizer = f"{study.priceNormalizer.mask}value"

        # Fetch the 'mask' value of the volumeNormalizer studyIndicator and append "value"
        volume_normalizer = f"{study.volumeNormalizer.mask}value"

        # print("Price_normalizer:", price_normalizer)
        # print("Volume_normalizer:", volume_normalizer)

        # Initialize the normalization map
        normalization_map = {
            "id": "NONE",
            "limitPrice": "PRICE",
            "takeProfit": "PRICE",
            "stopLoss": "PRICE",
            "direction": "DIRECTION",  # One-Hot Encoding later
            "status": "STATUS",  # Special case for status normalization
            "lpoffsetTP": "NONE",
            "slTP": "NONE",
            "tpTP": "NONE",
            "open": "PRICE",
            "close": "PRICE",
            "high": "PRICE",
            "low": "PRICE",
            "volume": "VOLUME"
        }

        # Add study indicators to the normalization map
        study_indicators = StudyIndicator.objects.filter(study=study)
        for indicator in study_indicators:
            normalization_map[f"{indicator.mask}value"] = indicator.indicator.normalizationType

        # Status mapping
        status_mapping = {
            "CLOSED_BY_SL": 0,
            "CLOSED_BY_TP": 1,
            "EXPIRED": 0
        }

        data = []
        for order in StudyOrder.objects.filter(study=study):
            order_data = {field.name: getattr(order, field.name) for field in StudyOrder._meta.fields}

            # Normalize the status field
            order_data["status"] = status_mapping.get(order_data["status"], order_data["status"])

            # Get the associated stock data item
            item = order.stockDataItem

            # Add the stock data item fields to the order data
            order_data.update({
                'open': float(item.open),
                'close': float(item.close),
                'high': float(item.high),
                'low': float(item.low),
                'volume': float(item.volume),
            })

            # Initialize normalizer values
            price_norm_value = None
            volume_norm_value = None

            # Handle indicator values         
            emptyIndicator = False
            indicator_values = StudyStockDataIndicatorValue.objects.filter(stockDataItem=item)
            for indicator_value in indicator_values:
                try:
                    indicator_data = json.loads(indicator_value.value)

                    # If value is empty, skip this iteration
                    if 'value' in indicator_data and (indicator_data['value'] is None or indicator_data['value'] != indicator_data['value']):
                        emptyIndicator = True
                        break  # Stop further processing if an empty indicator is found

                    for key, value in indicator_data.items():
                        if value is None or value != value:  # Check for NaN values
                            emptyIndicator = True
                            break
                        order_data.update({
                            f'{indicator_value.studyIndicator.mask}{key}': value
                        })
                    if emptyIndicator:
                        break
                except json.JSONDecodeError:
                    emptyIndicator = True
                    break  # Stop further processing if JSON decoding fails

            # Append the order data after processing all indicator values
            if not emptyIndicator:
                data.append(order_data)

        # Debug: Print collected data before creating DataFrame
        # print("Collected data:", data)

        # Create a DataFrame from the data
        df = pd.DataFrame(data)

        # Debug: Print DataFrame before excluding columns
        # print("DataFrame before excluding columns:", df)

        # Exclude specified fields
        df = df.drop(columns=['study', 'stockDataItem'], errors='ignore')
        df = df.drop(columns=['quantity', 'timeInForce', 'closedAt', 'createdAt', 'expiredAt', 'filledAt', 'cancelledAt', 'orderType'], errors='ignore')

        # Ensure normalizer columns are in the DataFrame
        if price_normalizer not in df.columns and price_norm_value is not None:
            df[price_normalizer] = price_norm_value
        if volume_normalizer not in df.columns and volume_norm_value is not None:
            df[volume_normalizer] = volume_norm_value

        # Debug: Print columns before normalization
        # print("Columns before normalization:", df.columns.tolist())

        # Convert all columns to float before normalization
        df = df.applymap(lambda x: float(x) if isinstance(x, Decimal) else x)

        # Apply normalization based on the normalization map
        for column, normalization_type in normalization_map.items():
            if normalization_type == 'PRICE':
                if price_normalizer in df.columns:
                    # Skip normalization for rows where the normalizer is NaN
                    df[column] = df.apply(
                        lambda row: row[column] / float(row[price_normalizer]) if not pd.isna(row[price_normalizer]) else row[column],
                        axis=1
                    )
                    print(f"Normalized {column} by {price_normalizer}")
                else:
                    print(f"Warning: price_normalizer column '{price_normalizer}' not found in DataFrame")
            elif normalization_type == 'VOLUME':
                if volume_normalizer in df.columns:
                    # Skip normalization for rows where the normalizer is NaN
                    df[column] = df.apply(
                        lambda row: row[column] / float(row[volume_normalizer]) if not pd.isna(row[volume_normalizer]) else row[column],
                        axis=1
                    )
                    print(f"Normalized {column} by {volume_normalizer}")
                else:
                    print(f"Warning: volume_normalizer column '{volume_normalizer}' not found in DataFrame")

        # One-Hot Encoding for direction
        direction_dummies = pd.get_dummies(df['direction'], prefix='', prefix_sep='')
        df = pd.concat([df.drop(columns=['direction']), direction_dummies], axis=1)

        # Convert True/False in BUY and SELL columns to 1/0
        df['BUY'] = df['BUY'].astype(int)
        df['SELL'] = df['SELL'].astype(int)

        # Replace infinities with NaNs
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop rows where all normalization resulted in NaNs
        df.dropna(how='all', inplace=True)

        # Drop normalizers columns
        df = df.drop(columns=[price_normalizer, volume_normalizer], errors='ignore')

        # Debug: Print columns before reordering
        # print("Columns before reordering:", df.columns.tolist())

        # Ensure DataFrame columns are unique before reordering
        if df.columns.duplicated().any():
            duplicated_columns = df.columns[df.columns.duplicated()].tolist()
            # print("Duplicated columns:", duplicated_columns)
            raise ValueError("DataFrame columns must be unique for orient='records'.")

        # Reorder columns to place BUY and SELL after id and status as last
        cols = df.columns.tolist()
        if 'status' in cols:
            cols.remove('status')
        if 'BUY' in cols:
            cols.remove('BUY')
        if 'SELL' in cols:
            cols.remove('SELL')
        cols.insert(cols.index('id') + 1, 'SELL')
        cols.insert(cols.index('id') + 1, 'BUY')
        cols.append('status')
        df = df[cols]

        # Debug: Print columns after reordering
        # print("Columns after reordering:", df.columns.tolist())

        # Save column order
        column_order = list(df.columns)

        # Convert DataFrame to JSON
        json_data = df.to_json(orient='records')

        # Convert JSON string to Python object
        data_object = json.loads(json_data)

        # Debug: Print data object before returning response
        # print("Data object:", data_object)

        # Include column order in the response
        response_data = {
            'data': data_object,
            'column_order': column_order
        }

        return self.create_response(request, response_data)

    # Get summary data for a study
    def get_summary_data(self, request, **kwargs):
        try:
            study = Study.objects.get(pk=kwargs['pk'])
        except Study.DoesNotExist:
            return self.create_response(request, {'error': 'not found'}, Http404)

        data = []
        for order in StudyOrder.objects.filter(study=study):
            order_data = {field.name: getattr(order, field.name) for field in StudyOrder._meta.fields}

            # Get the associated stock data item
            item = order.stockDataItem

            # Add the stock data item fields to the order data
            order_data.update({
                'open': item.open,
                'close': item.close,
                'high': item.high,
                'low': item.low,
                'volume': item.volume,
            })

            emptyIndicator = False
            indicator_values = StudyStockDataIndicatorValue.objects.filter(stockDataItem=item)
            for indicator_value in indicator_values:
                try:
                    indicator_data = json.loads(indicator_value.value)

                    # If value is empty, skip this iteration
                    if 'value' in indicator_data and (indicator_data['value'] is None or indicator_data['value'] != indicator_data['value']):
                        emptyIndicator = True
                        break  # Stop further processing if an empty indicator is found

                    for key, value in indicator_data.items():
                        if value is None or value != value:  # Check for NaN values
                            emptyIndicator = True
                            break
                        order_data.update({
                            f'{indicator_value.studyIndicator.mask}{key}': value
                        })
                    if emptyIndicator:
                        break
                except json.JSONDecodeError:
                    emptyIndicator = True
                    break  # Stop further processing if JSON decoding fails

            # Append the order data after processing all indicator values
            if not emptyIndicator:
                data.append(order_data)

        # Create a DataFrame from the data
        df = pd.DataFrame(data)

        # Exclude specified fields
        df = df.drop(columns=['study', 'stockDataItem'], errors='ignore')
        df = df.drop(columns=['quantity', 'timeInForce', 'closedAt', 'createdAt', 'expiredAt', 'filledAt', 'cancelledAt', 'orderType'], errors='ignore')

        # Save column order
        column_order = list(df.columns)

        # Convert DataFrame to JSON
        json_data = df.to_json(orient='records')

        # Convert JSON string to Python object
        data_object = json.loads(json_data)

        # Fetch the 'mask' value of the priceNormalizer studyIndicator
        price_normalizer = study.priceNormalizer.mask

        # Fetch the 'mask' value of the volumeNormalizer studyIndicator
        volume_normalizer = study.volumeNormalizer.mask

        # Include column order in the response
        response_data = {
            'data': data_object,
            'column_order': column_order,
            'priceNormalizer': price_normalizer,
            'volumeNormalizer': volume_normalizer
        }

        return self.create_response(request, response_data)

    # Get models for a study
    def get_study_models(self, request, **kwargs):

        try:
            study = Study.objects.get(pk=kwargs['pk'])
        except Study.DoesNotExist:
            return self.create_response(request, {'error': 'not found'}, Http404)

        models = NnModel.objects.filter(study=study)
        data = []
        for model in models:
            data.append({
                'id': model.id,
                'name': model.name,
                'description': model.description,
                'number_of_layers': model.number_of_layers,
                'nodes_per_layer': model.nodes_per_layer,
                'activation_function': model.activation_function,
                'loss_function': model.loss_function,
                'optimizer': model.optimizer,
                'learning_rate': model.learning_rate,
                'batch_size': model.batch_size,
                'number_of_epochs': model.number_of_epochs,
            })
                   
        return JsonResponse(data, safe=False)

    # Train model for a study
    def train_model(self, request, **kwargs):
         
        return train_nn_model(request, **kwargs)

    def check_status(self, request, **kwargs):
        return check_training_status(request, kwargs['model_id'])

    def get_trained_models(self, request, **kwargs):
        try:
            study_id = kwargs.get('pk')
            study = get_object_or_404(Study, pk=study_id)
        except Study.DoesNotExist:
            return self.create_response(request, {'error': 'Study not found'}, Http404)
        models = TrainedNnModel.objects.filter(study=study)
        data = []
        for model in models:
            data.append({
                'id': model.id,
                'nn_model_id': model.nn_model.id,
                'nn_model_name': model.nn_model.name,
                'serialized_model': model.serialized_model,
                'created_at': model.created_at,
            })
        return self.create_response(request, data)
    
    def get_trained_model_detail(self, request, **kwargs):
        # Extract `studyId` and `trainedModelId` from kwargs
        study_id = kwargs.get('studyId')
        trained_model_id = kwargs.get('trainedModelId')

        # Retrieve the specified study
        study = get_object_or_404(Study, pk=study_id)

        # Retrieve the specific trained model
        try:
            trained_model = TrainedNnModel.objects.get(study=study, id=trained_model_id)
        except TrainedNnModel.DoesNotExist:
            return self.create_response(request, {'error': 'Trained model not found'}, Http404)

        # Prepare response data
        data = {
            'id': trained_model.id,
            'nn_model_id': trained_model.nn_model.id,
            'nn_model_name': trained_model.nn_model.name,
            'serialized_model': trained_model.serialized_model,
            'created_at': trained_model.created_at,
        }

        return self.create_response(request, data)    
class IndicatorResource(ModelResource):
    class Meta:
        queryset = Indicator.objects.all()
        resource_name = 'indicators'
        allowed_methods = ['get']
class StudyIndicatorResource(ModelResource):
    class Meta:
        queryset = StudyIndicator.objects.all()
        resource_name = 'StudyIndicators'
        allowed_methods = ['get', 'delete', 'post']
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()
class StudyOrderResource(ModelResource):
    class Meta:
        queryset = StudyOrder.objects.all()
        resource_name = 'studyOrders'
        allowed_methods = ['get']
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()
class TradingPlanResource(ModelResource):
    class Meta:
        queryset = TradingPlan.objects.all()
        resource_name = 'tradingPlans'
        allowed_methods = ['get']
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()
class StudyTradingPlanResource(ModelResource):
    class Meta:
        queryset = StudyTradingPlan.objects.all()
        resource_name = 'studyTradingPlans'
        allowed_methods = ['get']
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()
class NnModelResource(ModelResource):
    class Meta:
        queryset = NnModel.objects.all()
        resource_name = 'nnModels'
        allowed_methods = ['get', 'delete', 'post']
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()

    def prepend_urls(self):
        return [
            # /api/nnModels/1/train/ - to train model with id 1
            re_path(r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/train%s$' % (self._meta.resource_name, trailing_slash()), self.wrap_view('train_model'), name="api_train_model"),
            # /api/nnModels/1/status/ - to check training status of model with id 1
            re_path(r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/status%s$' % (self._meta.resource_name, trailing_slash()), self.wrap_view('check_status'), name="api_check_status"),
        ]

    def train_model(self, request, **kwargs):
        model = get_object_or_404(NnModel, pk=kwargs['pk'])
        train_nn_model(model)
        return self.create_response(request, {'status': 'training started'})

    def check_status(self, request, **kwargs):
        model = get_object_or_404(NnModel, pk=kwargs['pk'])
        status = check_training_status(model)
        return self.create_response(request, {'status': status})
class TrainedNnModelResource(ModelResource):
    class Meta:
        queryset = TrainedNnModel.objects.all()
        resource_name = 'trainedNnModels'
        allowed_methods = ['get', 'post']
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()
class TradingSessionResource(ModelResource):
    class Meta:
        queryset = TradingSession.objects.all()
        resource_name = 'tradingSessions'
        allowed_methods = ['get', 'post', 'delete']  # add 'put' if you want
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()
        always_return_data = True
        filtering = {
            'type': ['exact'],
            'state': ['exact'],
            'status': ['exact'],
            'session_id': ['exact'],
            'name': ['icontains'],
            'createdAt': ['range', 'gte', 'lte'],
            'sessionStart': ['range', 'gte', 'lte', 'exact'],
            'sessionEnd': ['range', 'gte', 'lte', 'exact'],
        }

    def prepend_urls(self):
        return [
            # POST /api/v1/tradingSessions/<pk>/start/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/start%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('start_session'),
                name="api_trading_session_start",
            ),
            # POST /api/v1/tradingSessions/<pk>/stop/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/stop%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('stop_session'),
                name="api_trading_session_stop",
            ),
            # POST /api/v1/tradingSessions/<pk>/update_balance/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/update_balance%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('update_balance'),
                name="api_trading_session_update_balance",
            ),
            # POST /api/v1/tradingSessions/<pk>/save_indicators_bulk/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/save_indicators_bulk%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('save_indicators_bulk'),
                name="api_trading_session_save_indicators_bulk",
            ),
            # POST /api/v1/tradingSessions/<pk>/save_orders_bulk/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/save_orders_bulk%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('save_orders_bulk'),
                name="api_trading_session_save_orders_bulk",
            ),
        ]

    # ------- Custom endpoints (POST) -------

    def start_session(self, request, **kwargs):
        logger.info("start_session called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        session = get_object_or_404(TradingSession, pk=kwargs['pk'])
        # Simple state change; adjust as you implement runner logic
        session.state = 'RUNNING'
        session.status = 'OK'
        session.save(update_fields=['state', 'status', 'updatedAt'])

        self.log_throttled_access(request)
        return self.create_response(request, {'success': True, 'state': session.state, 'status': session.status})

    def stop_session(self, request, **kwargs):
        logger.info("stop_session called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        session = get_object_or_404(TradingSession, pk=kwargs['pk'])
        # You can choose COMPLETED vs ABORTED depending on payload (not shown)
        session.state = 'PAUSED'
        session.save(update_fields=['state', 'updatedAt'])

        self.log_throttled_access(request)
        return self.create_response(request, {'success': True, 'state': session.state})

    def update_balance(self, request, **kwargs):
        logger.info("update_balance called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        session = get_object_or_404(TradingSession, pk=kwargs['pk'])
        try:
            data = json.loads(request.body or "{}")
        except ValueError:
            raise ImmediateHttpResponse(HttpResponseBadRequest("Invalid JSON"))

        # Only update fields that are present in payload
        for f in [
            'InitialBalance', 'EquityNow', 'CashNow', 'BuyingPowerNow',
            'UnrealizedPnlNow', 'RealizedPnlTotal', 'MarginUsedNow', 'Currency'
        ]:
            if f in data:
                setattr(session, f, data[f])

        session.save()
        self.log_throttled_access(request)
        return self.create_response(request, {'success': True, 'balance': {
            'InitialBalance': str(session.InitialBalance),
            'EquityNow': str(session.EquityNow),
            'CashNow': str(session.CashNow),
            'BuyingPowerNow': str(session.BuyingPowerNow),
            'UnrealizedPnlNow': str(session.UnrealizedPnlNow),
            'RealizedPnlTotal': str(session.RealizedPnlTotal),
            'MarginUsedNow': str(session.MarginUsedNow),
            'Currency': session.Currency,
        }})

    def save_indicators_bulk(self, request, **kwargs):
        logger.info("save_indicators_bulk called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        session = get_object_or_404(TradingSession, pk=kwargs['pk'])
        try:
            data = json.loads(request.body or "{}")
        except ValueError:
            raise ImmediateHttpResponse(HttpResponseBadRequest("Invalid JSON"))

        # Expecting JSON like: {"indicators": {"sATR": [...], "VMA": [...]}}
        indicators = data.get('indicators')
        if indicators is None or not isinstance(indicators, dict):
            raise ImmediateHttpResponse(HttpResponseBadRequest("Missing or invalid 'indicators'"))

        # Merge into existing JSON bag
        current = session.sessionIndicatorsValues or {}
        current.update(indicators)
        session.sessionIndicatorsValues = current
        session.save(update_fields=['sessionIndicatorsValues', 'updatedAt'])

        self.log_throttled_access(request)
        return self.create_response(request, {'success': True})

    def save_orders_bulk(self, request, **kwargs):
        logger.info("save_orders_bulk called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        session = get_object_or_404(TradingSession, pk=kwargs['pk'])
        try:
            data = json.loads(request.body or "{}")
        except ValueError:
            raise ImmediateHttpResponse(HttpResponseBadRequest("Invalid JSON"))

        # Expecting JSON like: {"orders": [ {...}, {...} ]}
        orders = data.get('orders', [])
        if not isinstance(orders, list):
            raise ImmediateHttpResponse(HttpResponseBadRequest("Missing or invalid 'orders' array"))

        current = session.sessionOrders or {}
        # You can choose your structure; here we just append under a 'items' list
        current_items = current.get('items', [])
        current_items.extend(orders)
        current['items'] = current_items

        session.sessionOrders = current
        session.save(update_fields=['sessionOrders', 'updatedAt'])

        self.log_throttled_access(request)
        return self.create_response(request, {'success': True, 'added': len(orders)})
class SessionPotentialOrderResource(ModelResource):
    class Meta:
        queryset = SessionPotentialOrder.objects.all()
        resource_name = 'sessionPotentialOrders'
        allowed_methods = ['get', 'post', 'delete']  # add 'put' if you want updates
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()
        always_return_data = True
        filtering = {
            'session': ['exact'],
            'sessionStockDataItem': ['exact'],
            'trainedModel': ['exact'],
            'direction': ['exact'],
            'decision': ['exact'],
            'createdAt': ['range', 'gte', 'lte'],
        }

    # ---------- Custom endpoints ----------
    def prepend_urls(self):
        return [
            # POST /api/v1/sessionPotentialOrders/save_bulk/
            re_path(
                r'^sessionPotentialOrders/save_bulk%s$' % trailing_slash(),
                self.wrap_view('save_bulk'),
                name='api_session_potential_orders_save_bulk',
            ),
            # POST /api/v1/sessionPotentialOrders/<pk>/approve/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/approve%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('approve'),
                name='api_session_potential_order_approve',
            ),
            # POST /api/v1/sessionPotentialOrders/<pk>/reject/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/reject%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('reject'),
                name='api_session_potential_order_reject',
            ),
        ]

    # ---------- Bulk create ----------
    def save_bulk(self, request, *args, **kwargs):
        logger.info("SessionPotentialOrder.save_bulk called")

        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        try:
            payload = json.loads(request.body or "{}")
        except ValueError:
            raise ImmediateHttpResponse(HttpResponseBadRequest("Invalid JSON"))

        session_id = payload.get('sessionId')
        orders = payload.get('orders', [])

        if not session_id:
            raise ImmediateHttpResponse(HttpResponseBadRequest("Missing 'sessionId'"))
        if not isinstance(orders, list):
            raise ImmediateHttpResponse(HttpResponseBadRequest("'orders' must be a list"))

        session = get_object_or_404(TradingSession, pk=session_id)

        created = 0
        items = []
        for od in orders:
            # Expected keys in each order dict:
            # sessionStockDataItemId, trainedModelId, direction, limitPrice,
            # takeProfitPrice, stopPrice, lpOffset, slATR, tp, prediction, decision
            try:
                direction = od.get('direction')
                if direction not in ('BUY', 'SELL'):
                    raise ValueError("direction must be BUY or SELL")

                spdi = None
                if od.get('sessionStockDataItemId') is not None:
                    spdi = get_object_or_404(StockData, pk=od['sessionStockDataItemId'])

                tmodel = None
                if od.get('trainedModelId') is not None:
                    tmodel = get_object_or_404(TrainedNnModel, pk=od['trainedModelId'])

                def D(v):
                    return None if v in (None, '') else Decimal(str(v))

                spo = SessionPotentialOrder(
                    session=session,
                    sessionStockDataItem=spdi,
                    trainedModel=tmodel,
                    direction=direction,
                    limitPrice=D(od.get('limitPrice')),
                    takeProfitPrice=D(od.get('takeProfitPrice')),
                    stopPrice=D(od.get('stopPrice')),
                    lpOffset=D(od.get('lpOffset')),
                    slATR=D(od.get('slATR')),
                    tp=od.get('tp'),
                    prediction=od.get('prediction'),
                    decision=od.get('decision', 'NONE'),
                )
                # minimal validation mirrors model intent
                if spo.tp is not None and int(spo.tp) <= 0:
                    raise ValueError("tp must be a positive integer")
                spo.save()
                created += 1
                items.append(spo.id)
            except Exception as e:
                logger.exception("Failed to create SessionPotentialOrder: %s", e)
                # continue creating others; you can change to fail-fast if you prefer

        self.log_throttled_access(request)
        return self.create_response(
            request,
            {'success': True, 'created': created, 'ids': items, 'sessionId': session.id},
        )

    # ---------- Simple state helpers ----------
    def approve(self, request, **kwargs):
        logger.info("SessionPotentialOrder.approve called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        spo = get_object_or_404(SessionPotentialOrder, pk=kwargs['pk'])
        spo.decision = 'APPROVED'
        spo.save(update_fields=['decision', 'updatedAt'])

        self.log_throttled_access(request)
        return self.create_response(request, {'success': True, 'id': spo.id, 'decision': spo.decision})

    def reject(self, request, **kwargs):
        logger.info("SessionPotentialOrder.reject called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        spo = get_object_or_404(SessionPotentialOrder, pk=kwargs['pk'])
        spo.decision = 'REJECTED'
        spo.save(update_fields=['decision', 'updatedAt'])

        self.log_throttled_access(request)
        return self.create_response(request, {'success': True, 'id': spo.id, 'decision': spo.decision})
class SessionOrderResource(ModelResource):
    class Meta:
        queryset = SessionOrder.objects.all()
        resource_name = 'sessionOrders'
        allowed_methods = ['get', 'post', 'put', 'delete']
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()
        always_return_data = True
        filtering = {
            'session': ['exact'],
            'sessionPotentialOrder': ['exact'],
            'ticker': ['exact', 'icontains'],
            'status': ['exact'],
            'clientOrderId': ['exact'],
            'brokerOrderId': ['exact'],
            'createdAt': ['range', 'gte', 'lte'],
        }

    # ---------- URL extras ----------
    def prepend_urls(self):
        return [
            # POST /api/v1/sessionOrders/save_bulk/
            re_path(
                r'^sessionOrders/save_bulk%s$' % trailing_slash(),
                self.wrap_view('save_bulk'),
                name='api_session_orders_save_bulk',
            ),
            # POST /api/v1/sessionOrders/<pk>/cancel/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/cancel%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('cancel'),
                name='api_session_order_cancel',
            ),
            # POST /api/v1/sessionOrders/<pk>/ack/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/ack%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('acknowledge'),
                name='api_session_order_ack',
            ),
            # POST /api/v1/sessionOrders/<pk>/fill/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/fill%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('apply_fill'),
                name='api_session_order_fill',
            ),
            # POST /api/v1/sessionOrders/<pk>/expire/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/expire%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('expire'),
                name='api_session_order_expire',
            ),
        ]

    # ---------- Helpers ----------
    @staticmethod
    def D(v):
        return None if v in (None, '') else Decimal(str(v))

    def _validate_payload(self, od):
        # Basic enum checks (keep in sync with model)
        if od.get('side') not in ('BUY', 'SELL'):
            raise ImmediateHttpResponse(HttpResponseBadRequest("side must be BUY or SELL"))
        if od.get('orderType') not in ('MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT'):
            raise ImmediateHttpResponse(HttpResponseBadRequest("orderType invalid"))
        if od.get('timeInForce') and od['timeInForce'] not in ('GTC', 'IOC', 'FOK', 'DAY'):
            raise ImmediateHttpResponse(HttpResponseBadRequest("timeInForce invalid"))
        # Price requirements
        ot = od.get('orderType')
        if ot in ('LIMIT', 'STOP_LIMIT') and od.get('limitPrice') is None:
            raise ImmediateHttpResponse(HttpResponseBadRequest("limitPrice required for LIMIT/STOP_LIMIT"))
        if ot in ('STOP', 'STOP_LIMIT') and od.get('stopPrice') is None:
            raise ImmediateHttpResponse(HttpResponseBadRequest("stopPrice required for STOP/STOP_LIMIT"))
        # Quantity
        if od.get('quantity') in (None, ''):
            raise ImmediateHttpResponse(HttpResponseBadRequest("quantity required"))

    # ---------- Bulk create ----------
    def save_bulk(self, request, *args, **kwargs):
        logger.info("SessionOrder.save_bulk called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        try:
            payload = json.loads(request.body or "{}")
        except ValueError:
            raise ImmediateHttpResponse(HttpResponseBadRequest("Invalid JSON"))

        session_id = payload.get('sessionId')
        orders = payload.get('orders', [])
        if not session_id:
            raise ImmediateHttpResponse(HttpResponseBadRequest("Missing 'sessionId'"))
        if not isinstance(orders, list):
            raise ImmediateHttpResponse(HttpResponseBadRequest("'orders' must be a list"))

        session = get_object_or_404(TradingSession, pk=session_id)

        created, skipped, ids = 0, 0, []
        for od in orders:
            # Validate fields
            self._validate_payload(od)

            # Idempotency within session
            coid = od.get('clientOrderId')
            if not coid:
                raise ImmediateHttpResponse(HttpResponseBadRequest("clientOrderId is required for idempotency"))
            if SessionOrder.objects.filter(session=session, clientOrderId=coid).exists():
                skipped += 1
                continue

            spo = None
            if od.get('sessionPotentialOrderId') is not None:
                spo = get_object_or_404(SessionPotentialOrder, pk=od['sessionPotentialOrderId'])

            so = SessionOrder(
                session=session,
                sessionPotentialOrder=spo,
                parentOrder_id=od.get('parentOrderId'),
                ticker=od.get('ticker', ''),
                timeframe=od.get('timeframe'),
                venue=od.get('venue'),
                clientOrderId=coid,
                brokerOrderId=od.get('brokerOrderId'),
                ocoGroupId=od.get('ocoGroupId'),
                exitRole=od.get('exitRole', 'NONE'),

                side=od.get('side'),
                orderType=od.get('orderType'),
                timeInForce=od.get('timeInForce', 'GTC'),

                quantity=self.D(od.get('quantity')),
                limitPrice=self.D(od.get('limitPrice')),
                stopPrice=self.D(od.get('stopPrice')),
                reduceOnly=bool(od.get('reduceOnly', False)),
                postOnly=bool(od.get('postOnly', False)),
                expireAt=od.get('expireAt'),

                status='PLACED',
                createdAt=timezone.now(),  # keep created time here; placedAt set on ack
                rejectReason=None,

                filledQty=Decimal('0'),
                avgFillPrice=None,
                fees=Decimal('0'),
                slippage=self.D(od.get('slippage')),
                metadata=od.get('metadata'),
            )
            so.save()
            created += 1
            ids.append(so.id)

        self.log_throttled_access(request)
        return self.create_response(
            request,
            {'success': True, 'created': created, 'skipped': skipped, 'ids': ids, 'sessionId': session.id},
        )

    # ---------- Status transitions ----------
    def acknowledge(self, request, **kwargs):
        """Broker ack: set brokerOrderId and placedAt."""
        logger.info("SessionOrder.acknowledge called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        so = get_object_or_404(SessionOrder, pk=kwargs['pk'])
        try:
            payload = json.loads(request.body or "{}")
        except ValueError:
            raise ImmediateHttpResponse(HttpResponseBadRequest("Invalid JSON"))

        if payload.get('brokerOrderId'):
            so.brokerOrderId = payload['brokerOrderId']
        if not so.placedAt:
            so.placedAt = timezone.now()
        if so.status == 'PLACED':
            so.status = 'PLACED'  # remains placed; keeping for clarity

        so.save(update_fields=['brokerOrderId', 'placedAt', 'status', 'updatedAt'])
        self.log_throttled_access(request)
        return self.create_response(request, {'success': True, 'id': so.id, 'brokerOrderId': so.brokerOrderId})

    def apply_fill(self, request, **kwargs):
        """Apply a (partial) fill; updates filledQty, avgFillPrice, fees, status."""
        logger.info("SessionOrder.apply_fill called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        so = get_object_or_404(SessionOrder, pk=kwargs['pk'])
        try:
            payload = json.loads(request.body or "{}")
        except ValueError:
            raise ImmediateHttpResponse(HttpResponseBadRequest("Invalid JSON"))

        qty = payload.get('qty')
        price = payload.get('price')
        fee = payload.get('fee', 0)

        if qty in (None, '') or price in (None, ''):
            raise ImmediateHttpResponse(HttpResponseBadRequest("qty and price are required"))

        qty = Decimal(str(qty))
        price = Decimal(str(price))
        fee = Decimal(str(fee))

        # VWAP update
        prev_qty = so.filledQty or Decimal('0')
        prev_value = (so.avgFillPrice or Decimal('0')) * prev_qty
        new_qty = prev_qty + qty
        new_value = prev_value + (price * qty)

        so.filledQty = new_qty
        so.avgFillPrice = (new_value / new_qty) if new_qty > 0 else None
        so.fees = (so.fees or Decimal('0')) + fee

        # Status transitions
        if so.quantity and new_qty >= so.quantity:
            so.status = 'FILLED'
            if not so.filledAt:
                so.filledAt = timezone.now()
        else:
            so.status = 'PARTIAL'

        so.save(update_fields=['filledQty', 'avgFillPrice', 'fees', 'status', 'filledAt', 'updatedAt'])
        self.log_throttled_access(request)
        return self.create_response(request, {
            'success': True,
            'id': so.id,
            'filledQty': str(so.filledQty),
            'avgFillPrice': str(so.avgFillPrice) if so.avgFillPrice is not None else None,
            'fees': str(so.fees),
            'status': so.status,
        })

    def cancel(self, request, **kwargs):
        logger.info("SessionOrder.cancel called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        so = get_object_or_404(SessionOrder, pk=kwargs['pk'])
        so.status = 'CANCELLED'
        so.cancelledAt = timezone.now()
        so.save(update_fields=['status', 'cancelledAt', 'updatedAt'])

        self.log_throttled_access(request)
        return self.create_response(request, {'success': True, 'id': so.id, 'status': so.status})

    def expire(self, request, **kwargs):
        logger.info("SessionOrder.expire called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        so = get_object_or_404(SessionOrder, pk=kwargs['pk'])
        so.status = 'EXPIRED'
        so.expiredAt = timezone.now()
        so.save(update_fields=['status', 'expiredAt', 'updatedAt'])

        self.log_throttled_access(request)
        return self.create_response(request, {'success': True, 'id': so.id, 'status': so.status})
class SessionFillResource(ModelResource):
    class Meta:
        queryset = SessionFill.objects.all()
        resource_name = 'sessionFills'
        allowed_methods = ['get', 'post', 'delete']  # add 'put' if you plan to edit fills
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()
        always_return_data = True
        filtering = {
            'session': ['exact'],
            'sessionOrder': ['exact'],
            'brokerTradeId': ['exact'],
            'ts': ['range', 'gte', 'lte'],
        }

    # --------- URLs ----------
    def prepend_urls(self):
        return [
            # POST /api/v1/sessionFills/save_bulk/
            re_path(
                r'^sessionFills/save_bulk%s$' % trailing_slash(),
                self.wrap_view('save_bulk'),
                name='api_session_fills_save_bulk',
            ),
        ]

    # --------- helpers ----------
    @staticmethod
    def D(v):
        return None if v in (None, '') else Decimal(str(v))

    def _recalc_order_rollups(self, so: SessionOrder):
        """Recalculate filledQty, avgFillPrice, fees, status, filledAt from all fills."""
        fills = list(so.fills.order_by('ts', 'id').values('qty', 'price', 'fee'))
        total_qty = sum((f['qty'] for f in fills), Decimal('0'))
        total_value = sum((Decimal(f['qty']) * Decimal(f['price']) for f in fills), Decimal('0'))
        total_fees = sum((f['fee'] for f in fills), Decimal('0'))

        so.filledQty = total_qty
        so.avgFillPrice = (total_value / total_qty) if total_qty > 0 else None
        so.fees = total_fees

        # Status transitions
        if so.quantity is not None and total_qty >= so.quantity:
            so.status = 'FILLED'
            if not so.filledAt:
                so.filledAt = timezone.now()
        elif total_qty > 0:
            so.status = 'PARTIAL'
        # else keep existing status (PLACED/CANCELLED/EXPIRED/REJECTED)

        so.save(update_fields=['filledQty', 'avgFillPrice', 'fees', 'status', 'filledAt', 'updatedAt'])

    # --------- bulk create ----------
    def save_bulk(self, request, *args, **kwargs):
        """
        Payload examples:

        1) Common session + order:
        {
          "sessionId": 42,
          "sessionOrderId": 101,
          "fills": [
            {"ts": "2025-10-31T08:12:00Z", "qty": 0.5, "price": 100.01, "fee": 0.0005, "liquidityType": "MAKER", "brokerTradeId": "T1"},
            {"ts": "2025-10-31T08:13:00Z", "qty": 0.5, "price": 99.99,  "fee": 0.0005, "liquidityType": "TAKER", "brokerTradeId": "T2"}
          ]
        }

        2) Mixed orders per fill:
        {
          "sessionId": 42,
          "fills": [
            {"sessionOrderId": 101, "ts": "...", "qty": 0.1, "price": 10.0},
            {"sessionOrderId": 102, "ts": "...", "qty": 0.2, "price": 10.1}
          ]
        }
        """
        logger.info("SessionFill.save_bulk called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        try:
            payload = json.loads(request.body or "{}")
        except ValueError:
            raise ImmediateHttpResponse(HttpResponseBadRequest("Invalid JSON"))

        session_id = payload.get('sessionId')
        if not session_id:
            raise ImmediateHttpResponse(HttpResponseBadRequest("Missing 'sessionId'"))
        session = get_object_or_404(TradingSession, pk=session_id)

        default_order_id = payload.get('sessionOrderId')
        fills = payload.get('fills', [])
        if not isinstance(fills, list):
            raise ImmediateHttpResponse(HttpResponseBadRequest("'fills' must be a list"))

        created, duplicates = 0, 0
        touched_orders = set()

        for f in fills:
            order_id = f.get('sessionOrderId', default_order_id)
            if not order_id:
                raise ImmediateHttpResponse(HttpResponseBadRequest("Each fill must provide 'sessionOrderId' or top-level 'sessionOrderId'"))

            so = get_object_or_404(SessionOrder, pk=order_id, session=session)

            qty = self.D(f.get('qty'))
            price = self.D(f.get('price'))
            if qty in (None, '') or price in (None, ''):
                raise ImmediateHttpResponse(HttpResponseBadRequest("Each fill requires 'qty' and 'price'"))

            fee = self.D(f.get('fee')) or Decimal('0')
            ts = f.get('ts') or timezone.now()
            liquidity = f.get('liquidityType')  # optional: "MAKER"/"TAKER"/None
            broker_trade_id = f.get('brokerTradeId')

            try:
                sf = SessionFill.objects.create(
                    session=session,
                    sessionOrder=so,
                    ts=ts,
                    qty=qty,
                    price=price,
                    fee=fee,
                    liquidityType=liquidity,
                    brokerTradeId=broker_trade_id,
                    metadata=f.get('metadata'),
                )
                created += 1
                touched_orders.add(so.id)
            except IntegrityError:
                # Likely the unique constraint (sessionOrder, brokerTradeId) hit for duplicates
                duplicates += 1
                continue

        # Recalculate roll-ups once per touched order
        for oid in touched_orders:
            so = SessionOrder.objects.get(pk=oid)
            self._recalc_order_rollups(so)

        self.log_throttled_access(request)
        return self.create_response(
            request,
            {'success': True, 'created': created, 'duplicates': duplicates, 'touchedOrders': list(touched_orders)}
        )
class SessionStockDataResource(ModelResource):
    class Meta:
        queryset = SessionStockData.objects.all()
        resource_name = 'SessionStockData'
        allowed_methods = ['get', 'delete', 'post']
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()

    def prepend_urls(self):
        return [
            re_path(r'^SessionStockData/save_bulk/$', self.wrap_view('save_bulk'), name='api_session_save_bulk'),
        ]

    def save_bulk(self, request, *args, **kwargs):
        logger.info("SessionStockDataResource.save_bulk called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        try:
            payload = json.loads(request.body or "{}")
        except json.JSONDecodeError:
            return self.error_response(request, {"success": False, "message": "Invalid JSON"}, response_class=http.HttpBadRequest)

        session_id = payload.get('sessionId')
        ticker = payload.get('ticker')
        timeframe = payload.get('timeFrame')
        results = payload.get('results', [])

        if not session_id or not ticker or not timeframe or not isinstance(results, list):
            return self.error_response(
                request,
                {"success": False, "message": "Missing or invalid 'sessionId', 'ticker', 'timeFrame', or 'results'"},
                response_class=http.HttpBadRequest
            )

        trading_session = get_object_or_404(TradingSession, id=session_id)

        created, updated = 0, 0
        with transaction.atomic():
            for item in results:
                t = int(item.get('t'))
                defaults = {
                    'volume': int(item.get('v', 0)),
                    'vw': float(item.get('vw', 0.0)),
                    'open': float(item.get('o', 0.0)),
                    'close': float(item.get('c', 0.0)),
                    'high': float(item.get('h', 0.0)),
                    'low': float(item.get('l', 0.0)),
                    'transactions': int(item.get('n', 0)),
                    'timeframe': timeframe,
                }
                obj, was_created = SessionStockData.objects.update_or_create(
                    trading_session=trading_session,
                    ticker=ticker,
                    timestamp=t,
                    defaults=defaults
                )
                created += int(was_created)
                updated += int(not was_created)

        self.log_throttled_access(request)
        return self.create_response(request, {
            'success': True,
            'message': 'Session stock data saved',
            'created': created,
            'updated': updated
        })

class CourseResource(ModelResource):
    class Meta:
        queryset = Course.objects.all()
        resource_name = 'courses'
        allowed_methods = ['get', 'delete', 'post']
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()

    def hydrate(self, bundle):
        bundle.obj.category_id = bundle.data['category_id']
        return bundle
    
    def dehydrate(self, bundle):
        bundle.data['category_id'] = bundle.obj.category_id
        bundle.data['category'] = bundle.obj.category
        return bundle
    

    
