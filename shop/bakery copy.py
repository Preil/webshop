import json
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback
from shop.models import NnModel, Study, StudyOrder, StudyStockDataIndicatorValue, StudyIndicator
import numpy as np
from decimal import Decimal
from django.http import JsonResponse, Http404
import threading

# Initialize the global training status dictionary
training_status = {}

# Define a callback class to store the epoch, loss, and accuracy values
class TrainingProgressCallback(Callback):
    def __init__(self, model_id):
        super().__init__()
        self.model_id = model_id

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            training_status[self.model_id] = {
                "status": "Training",
                "epoch": epoch,
                "loss": logs.get("loss"),
                "accuracy": logs.get("accuracy")
            }

# Define the train_model function
def train_model_with_status(data, labels, model_params, model_id):
    model = Sequential()
    nodes_per_layer = list(map(int, model_params.nodes_per_layer.split(',')))
    for nodes in nodes_per_layer:
        model.add(Dense(nodes, activation=model_params.activation_function))
    model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification, adjust as needed
    model.compile(optimizer=model_params.optimizer, loss=model_params.loss_function, metrics=['accuracy'])

    def train():
        global training_status
        training_status[model_id] = {"status": "Training"}

        progress_callback = TrainingProgressCallback(model_id)

        model.fit(data, labels, epochs=model_params.number_of_epochs, batch_size=model_params.batch_size, callbacks=[progress_callback])
        model_path = f"models/{model_params.name}.h5"
        model.save(model_path)
        training_status[model_id] = {"status": "Completed", "model_path": model_path}

    threading.Thread(target=train).start()
    return None

# Define the train_nn_model function
def train_nn_model(request, **kwargs):
    try:
        model = NnModel.objects.get(pk=kwargs['model_id'])
        study = Study.objects.get(pk=kwargs['pk'])
        target_column = 'status'  # Replace with your actual target column

        normalized_data_response = get_normalized_data(study, target_column)
        data = pd.DataFrame(normalized_data_response['data'])
        labels = data.pop(target_column)  # Extract labels from the target column

        global training_status
        training_status[model.id] = {"status": "Not started"}

        train_model_with_status(data.values, labels.values, model, model.id)

        return JsonResponse({'message': 'Model training started'})
    except NnModel.DoesNotExist:
        return JsonResponse({"error": "Model not found"}, status=404)
    except Study.DoesNotExist:
        return JsonResponse({"error": "Study not found"}, status=404)
    except KeyError as e:
        return JsonResponse({"error": f"Missing key: {str(e)}"}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

def check_training_status(request, model_id):
    global training_status
    status = training_status.get(int(model_id), {"status": "Not started"})
    return JsonResponse(status)

# Define the get_normalized_data function
def get_normalized_data(study, target_column):
    def get_indicator_normalization_type(indicator_mask):
        indicators = StudyIndicator.objects.filter(mask=indicator_mask)
        if indicators.count() == 1:
            return indicators.first().indicator.normalizationType
        else:
            raise ValueError(f"Multiple or no StudyIndicator objects found for mask '{indicator_mask}'")

    price_normalizer = f"{study.priceNormalizer.mask}value"
    volume_normalizer = f"{study.volumeNormalizer.mask}value"

    normalization_map = {
        "id": "NONE",
        "limitPrice": "PRICE",
        "takeProfit": "PRICE",
        "stopLoss": "PRICE",
        "direction": "DIRECTION",
        "status": "STATUS",
        "lpoffsetTP": "NONE",
        "slTP": "NONE",
        "tpTP": "NONE",
        "open": "PRICE",
        "close": "PRICE",
        "high": "PRICE",
        "low": "PRICE",
        "volume": "VOLUME"
    }

    study_indicators = StudyIndicator.objects.filter(study=study)
    for indicator in study_indicators:
        normalization_map[f"{indicator.mask}value"] = indicator.indicator.normalizationType

    status_mapping = {
        "CLOSED_BY_SL": 0,
        "CLOSED_BY_TP": 1,
        "EXPIRED": 0
    }

    data = []
    for order in StudyOrder.objects.filter(study=study):
        order_data = {field.name: getattr(order, field.name) for field in StudyOrder._meta.fields}
        order_data["status"] = status_mapping.get(order_data["status"], order_data["status"])
        item = order.stockDataItem
        order_data.update({
            'open': float(item.open),
            'close': float(item.close),
            'high': float(item.high),
            'low': float(item.low),
            'volume': float(item.volume),
        })

        emptyIndicator = False
        indicator_values = StudyStockDataIndicatorValue.objects.filter(stockDataItem=item)
        for indicator_value in indicator_values:
            try:
                indicator_data = json.loads(indicator_value.value)
                if 'value' in indicator_data and (indicator_data['value'] is None or indicator_data['value'] != indicator_data['value']):
                    emptyIndicator = True
                    break
                for key, value in indicator_data.items():
                    if value is None or value != value:
                        emptyIndicator = True
                        break
                    order_data.update({
                        f'{indicator_value.studyIndicator.mask}{key}': value
                    })
                if emptyIndicator:
                    break
            except json.JSONDecodeError:
                emptyIndicator = True
                break

        if not emptyIndicator:
            data.append(order_data)

    df = pd.DataFrame(data)
    df = df.drop(columns=['study', 'stockDataItem'], errors='ignore')
    df = df.drop(columns=['quantity', 'timeInForce', 'closedAt', 'createdAt', 'expiredAt', 'filledAt', 'cancelledAt', 'orderType'], errors='ignore')

    if price_normalizer not in df.columns:
        price_norm_value = 1  # Assuming a default value of 1 if not found
        df[price_normalizer] = price_norm_value
    if volume_normalizer not in df.columns:
        volume_norm_value = 1  # Assuming a default value of 1 if not found
        df[volume_normalizer] = volume_norm_value

    df = df.applymap(lambda x: float(x) if isinstance(x, Decimal) else x)

    for column, normalization_type in normalization_map.items():
        if normalization_type == 'PRICE':
            if price_normalizer in df.columns:
                df[column] = df.apply(
                    lambda row: row[column] / float(row[price_normalizer]) if not pd.isna(row[price_normalizer]) else row[column],
                    axis=1
                )
        elif normalization_type == 'VOLUME':
            if volume_normalizer in df.columns:
                df[column] = df.apply(
                    lambda row: row[column] / float(row[volume_normalizer]) if not pd.isna(row[volume_normalizer]) else row[column],
                    axis=1
                )

    direction_dummies = pd.get_dummies(df['direction'], prefix='', prefix_sep='')
    df = pd.concat([df.drop(columns=['direction']), direction_dummies], axis=1)
    df['BUY'] = df['BUY'].astype(int)
    df['SELL'] = df['SELL'].astype(int)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(how='all', inplace=True)
    df = df.drop(columns=[price_normalizer, volume_normalizer], errors='ignore')

    if df.columns.duplicated().any():
        duplicated_columns = df.columns[df.columns.duplicated()].tolist()
        raise ValueError("DataFrame columns must be unique for orient='records'.")

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

    column_order = list(df.columns)
    data_object = df.to_dict(orient='records')

    return {
        'data': data_object,
        'column_order': column_order
    }
