import json
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.models import model_from_json
from shop.models import TrainedNnModel, NnModel, Study, StudyOrder, StudyStockDataIndicatorValue, StudyIndicator
import numpy as np
from decimal import Decimal
from django.http import JsonResponse, Http404
import threading
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import base64

# Initialize the global training status dictionary
training_status = {}

# Mapping for optimizers
OPTIMIZER_MAPPING = {
    'sgd': SGD,
    'adam': Adam,
    'rmsprop': RMSprop,
}

def balanced_batch_generator(data, labels, batch_size):
    if batch_size % 2 != 0:
        raise ValueError("Batch size must be even for balanced batch generation")

    class_0 = np.array([x for x, y in zip(data, labels) if y == 0])
    class_1 = np.array([x for x, y in zip(data, labels) if y == 1])

    min_class_samples = min(len(class_0), len(class_1))
    batches_per_epoch = min_class_samples // (batch_size // 2)

    while True:
        class_0 = shuffle(class_0)
        class_1 = shuffle(class_1)

        for i in range(batches_per_epoch):
            start_idx = i * (batch_size // 2)
            end_idx = (i + 1) * (batch_size // 2)

            batch_data = np.concatenate((class_0[start_idx:end_idx], class_1[start_idx:end_idx]), axis=0)
            batch_labels = np.array([0] * (batch_size // 2) + [1] * (batch_size // 2))

            yield shuffle(batch_data, batch_labels)

        remaining_class_0 = class_0[batches_per_epoch * (batch_size // 2):]
        remaining_class_1 = class_1[batches_per_epoch * (batch_size // 2):]

        remaining_batch_data = np.concatenate((remaining_class_0, remaining_class_1), axis=0)
        remaining_batch_labels = np.array([0] * len(remaining_class_0) + [1] * len(remaining_class_1))

        if len(remaining_batch_data) > 0:
            yield shuffle(remaining_batch_data, remaining_batch_labels)

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

def train_model_with_status(data, labels, model_params, model_id, study_id):
    model = Sequential()
    input_dim = data.shape[1]
    nodes_per_layer = list(map(int, model_params.nodes_per_layer.split(',')))
    for nodes in nodes_per_layer:
        model.add(Dense(nodes, activation=model_params.activation_function, input_dim=input_dim))
        model.add(Dropout(0.5))
        input_dim = None

    final_activation = 'sigmoid' if model_params.activation_function not in ['softmax'] else model_params.activation_function
    model.add(Dense(1, activation=final_activation))

    optimizer_class = OPTIMIZER_MAPPING.get(model_params.optimizer)
    if optimizer_class:
        optimizer = optimizer_class(learning_rate=model_params.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {model_params.optimizer}")

    model.compile(optimizer=optimizer, loss=model_params.loss_function, metrics=['accuracy'])

    def train():
        global training_status
        training_status[model_id] = {"status": "Training"}
    
        progress_callback = TrainingProgressCallback(model_id)
    
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)
    
        # Use balanced batch generator for training data
        train_generator = balanced_batch_generator(X_train, y_train, batch_size=model_params.batch_size)
        steps_per_epoch = min(len([y for y in y_train if y == 0]), len([y for y in y_train if y == 1])) // (model_params.batch_size // 2)
    
        # Early stopping and learning rate reduction callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    
        # Train model
        model.fit(train_generator,
                  epochs=model_params.number_of_epochs,
                  steps_per_epoch=steps_per_epoch,
                  validation_data=(X_val, y_val),
                  callbacks=[progress_callback, early_stopping, reduce_lr])
    
        # Save the model to the database
        save_model_to_db(model, model_id, study_id)
        training_status[model_id] = {"status": "Completed"}

    threading.Thread(target=train).start()
    return None

def train_nn_model(request, **kwargs):
    try:
        model = NnModel.objects.get(pk=kwargs['model_id'])
        study = Study.objects.get(pk=kwargs['pk'])

        normalized_data_response = get_normalized_data(study, target_column='status')
        data = pd.DataFrame(normalized_data_response['data'])
        labels = data.pop('status')
        # 🔍 Debug: verify columns used for training
        print("TRAINING FEATURE COLUMNS:", list(data.columns))
        print("NUMBER OF FEATURES:", len(data.columns))
        
        train_model_with_status(data.values, labels.values, model, model.id, study.id)
        return JsonResponse({'message': 'Model training started'})
    except NnModel.DoesNotExist:
        return JsonResponse({"error": "Model not found"}, status=404)
    except Study.DoesNotExist:
        return JsonResponse({"error": "Study not found"}, status=404)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

def check_training_status(request, model_id):
    global training_status
    status = training_status.get(int(model_id), {"status": "Not started"})
    return JsonResponse(status)

def save_model_to_db(model, nn_model_id, study_id):
    serialized_model = model.to_json()
    encoded_model = base64.b64encode(serialized_model.encode('utf-8'))  # Encode as bytes

    trained_nn_model = TrainedNnModel(
        nn_model_id=nn_model_id,
        study_id=study_id,
        serialized_model=encoded_model  # Save as bytes
    )
    trained_nn_model.save()

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
    df = df.drop(columns=['quantity', 'timeInForce', 'closedAt', 'createdAt', 'expiredAt', 'filledAt', 'cancelledAt', 'orderType', 'id'], errors='ignore')

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

    ordered_cols = []

    # Same order as in _build_nn_input_from_raw
    if 'BUY' in df.columns:
        ordered_cols.append('BUY')
    if 'SELL' in df.columns:
        ordered_cols.append('SELL')

    # Add remaining feature columns
    for c in cols:
        if c not in ordered_cols:
            ordered_cols.append(c)

    # Target column last
    ordered_cols.append('status')

    df = df[ordered_cols]

    column_order = list(df.columns)
    data_object = df.to_dict(orient='records')

    return {
        'data': data_object,
        'column_order': column_order
    }
