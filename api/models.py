from django.shortcuts import get_object_or_404
from django.core import serializers
from tastypie.resources import ModelResource
from tastypie.utils import trailing_slash
from shop.models import Category, Course, StockData, Study, Indicator, StudyIndicator, StudyStockDataIndicatorValue, StudyOrder, TradingPlan, StudyTradingPlan
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
logger = logging.getLogger(__name__)

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
    def get_normalized_data_4(self, request, **kwargs):
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

        print("Price_normalizer:", price_normalizer)
        print("Volume_normalizer:", volume_normalizer)

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

        # Replace infinities with NaNs
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop rows where all normalization resulted in NaNs
        df.dropna(how='all', inplace=True)

        # Drop normalizers columns
        df = df.drop(columns=[price_normalizer, volume_normalizer], errors='ignore')

        # Reorder columns to place BUY and SELL after id and status as last
        cols = df.columns.tolist()
        cols.remove('status')
        cols.insert(cols.index('id') + 1, 'SELL')
        cols.insert(cols.index('id') + 1, 'BUY')
        cols.append('status')
        df = df[cols]

        # Debug: Print DataFrame after replacing infinities and NaNs
        # print("DataFrame after replacing infinities and NaNs:", df)

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



    # Get normalized data for a study
    def get_normalized_data_old2(self, request, **kwargs):
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

        print("Price_normalizer:", price_normalizer)
        print("Volume_normalizer:", volume_normalizer)

        # Initialize the normalization map
        normalization_map = {
            "id": "NONE",
            "limitPrice": "PRICE",
            "takeProfit": "PRICE",
            "stopLoss": "PRICE",
            "direction": "NONE",
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

        # Debug: Print DataFrame after normalization
        # print("DataFrame after normalization:", df)

        # Replace infinities with NaNs
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop rows where all normalization resulted in NaNs
        df.dropna(how='all', inplace=True)

        # Drop normalizers columns
        df = df.drop(columns=[price_normalizer, volume_normalizer], errors='ignore')

        # Debug: Print DataFrame after replacing infinities and NaNs
        # print("DataFrame after replacing infinities and NaNs:", df)

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


    # Get normalized data for a study
    def get_normalized_data_old(self, request, **kwargs):
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

        print("Price_normalizer:", price_normalizer)
        print("Volume_normalizer:", volume_normalizer)

        # Initialize the normalization map
        normalization_map = {
            "id": "NONE",
            "limitPrice": "PRICE",
            "takeProfit": "PRICE",
            "stopLoss": "PRICE",
            "direction": "NONE",
            "status": "NONE",
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

        data = []
        for order in StudyOrder.objects.filter(study=study):
            order_data = {field.name: getattr(order, field.name) for field in StudyOrder._meta.fields}

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
        #print("Collected data:", data)

        # Create a DataFrame from the data
        df = pd.DataFrame(data)

        # Debug: Print DataFrame before excluding columns
        #print("DataFrame before excluding columns:", df)

        # Exclude specified fields
        df = df.drop(columns=['study', 'stockDataItem'], errors='ignore')
        df = df.drop(columns=['quantity', 'timeInForce', 'closedAt', 'createdAt', 'expiredAt', 'filledAt', 'cancelledAt', 'orderType'], errors='ignore')

        # Ensure normalizer columns are in the DataFrame
        if price_normalizer not in df.columns and price_norm_value is not None:
            df[price_normalizer] = price_norm_value
        if volume_normalizer not in df.columns and volume_norm_value is not None:
            df[volume_normalizer] = volume_norm_value

        # Debug: Print columns before normalization
        #print("Columns before normalization:", df.columns.tolist())

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

        # Debug: Print DataFrame after normalization
        # print("DataFrame after normalization:", df)

        # Replace infinities with NaNs
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop rows where all normalization resulted in NaNs
        df.dropna(how='all', inplace=True)
        
        # Drop normalizers columns
        df = df.drop(columns=[price_normalizer, volume_normalizer], errors='ignore')

        # Debug: Print DataFrame after replacing infinities and NaNs
        # print("DataFrame after replacing infinities and NaNs:", df)

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
    

    
