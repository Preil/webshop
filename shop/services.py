# services.py
from .models import Study, StockData, StudyIndicator, StudyOrder, StudyStockDataIndicatorValue, StudyTradingPlan
from django.core.exceptions import ObjectDoesNotExist
from . import calculations
import json
import math
from datetime import datetime
from .session_models import SessionStockData, SessionStockDataIndicatorValue
from django.db import transaction
from .session_models import SessionStockData, SessionStockDataIndicatorValue

INDICATOR_FUNCTIONS = {
    '#movingAverage': calculations.movingAverage,
    '#atr': calculations.atr,
    '#rsi': calculations.rsi,
    '#satr': calculations.satr,
    '#vma': calculations.vma, # Volume weighted Price Moving Average
    '#va' : calculations.va, # Volume Average
    '#va_slope' : calculations.va_slope, # Volume Average Slope
    '#ma_slope' : calculations.ma_slope, # Moving Average Slope
    # Add more indicators as needed...
}

def calculate_something(data, indicator_name):
    # Get the appropriate function
    calculate_indicator = INDICATOR_FUNCTIONS[indicator_name]

    # Call the function and return the result
    return calculate_indicator(data)


def calculateStudy(study):
    # Retrieve the StockData for the study
    stock_data = StockData.objects.filter(study=study)

    # Retrieve the StudyIndicators for the study
    study_indicators = StudyIndicator.objects.filter(study=study)
    
    # Delete all existing StudyStockDataIndicatorValue records for these study indicators
    StudyStockDataIndicatorValue.objects.filter(studyIndicator__in=study_indicators).delete()
    
    # Get the functionName for each StudyIndicator
    for indicator in study_indicators:
        indicator_function_name = indicator.indicator.functionName
        function = INDICATOR_FUNCTIONS[indicator_function_name]
        # Perform calculations...
        print("Call function: ", function)
        results = function(stock_data, indicator.parametersValue, indicator.id)
               
    
        
        # Save the results to the StudyStockDataIndicatorValue model
        for study_indicator_id, values in results.items():
            study_indicator = StudyIndicator.objects.get(id=study_indicator_id)
            for id, value in values.items():
                try:
                    stock_data_item = StockData.objects.get(id=id)
                    StudyStockDataIndicatorValue.objects.create(studyIndicator=study_indicator, stockDataItem=stock_data_item, value=value)
                except ObjectDoesNotExist:
                    print(f"StockData with id {id} does not exist")

    return "Results calculated and saved to the database"

def calculateSession(session):
    """Calculate indicator values for a TradingSession (session tables)."""
    study = getattr(session, 'study', None)
    if not study:
        return {"ok": False, "message": "Session has no linked Study"}

    # Bars for this session
    stock_data_qs = SessionStockData.objects.filter(trading_session=session).order_by('timestamp')

    # Indicators (mask/function/params) from the linked study
    sis = StudyIndicator.objects.filter(study=study).select_related('indicator').order_by('id')

    # Clear previous values for these indicators within THIS session
    SessionStockDataIndicatorValue.objects.filter(
        sessionStockDataItem__trading_session=session,
        studyIndicator__in=sis
    ).delete()

    created, updated = 0, 0

    # Compute & persist
    with transaction.atomic():
        for si in sis:
            fn_name = getattr(si.indicator, "functionName", None)
            fn = INDICATOR_FUNCTIONS.get(fn_name)
            if not fn:
                continue

            # Your calc fns expect (qs, params_json_string, study_indicator_id)
            results = fn(stock_data_qs, si.parametersValue, si.id)
            if not results:
                continue

            # results: { study_indicator_id: { stockdata_id: json_string, ... } }
            for study_indicator_id, per_item in results.items():
                # avoid extra lookups if misaligned
                if study_indicator_id != si.id:
                    try:
                        si_ref = StudyIndicator.objects.get(id=study_indicator_id)
                    except ObjectDoesNotExist:
                        continue
                else:
                    si_ref = si

                for sd_id, json_value in per_item.items():
                    try:
                        sdi = SessionStockData.objects.get(id=sd_id)
                    except ObjectDoesNotExist:
                        continue

                    # Only fields that actually exist on the model:
                    SessionStockDataIndicatorValue.objects.create(
                        studyIndicator=si_ref,
                        sessionStockDataItem=sdi,
                        value=json_value  # keep same JSON string format as Study
                    )
                    created += 1

    return {"ok": True, "created": created, "updated": updated, "indicators": sis.count()}

    
# Simulate trades with trading plan
def simulate_trades(study,tradingPlanParams):
    print("Simulate trades with trading plan")

    # Check that Study has mainSatrIndicator set
    # if not getattr(study, "mainSatrIndicator_id", None):
    #     return {"ok": False, "message": "Set Study.mainSatrIndicator to an exact StudyIndicator (e.g., sATR14) before simulating."}
    # Parse tradingPlanParams to a dictionary
    tradingPlanParams = json.loads(tradingPlanParams)

    # Delete all existing StudyOrder records for the study
    StudyOrder.objects.filter(study=study).delete()
    
    # Get study stockData
    studyStockData = StockData.objects.filter(study=study)
    # Ensure stock data is sorted by timestamp in ascending order
    studyStockData = studyStockData.order_by('timestamp')

    # Set expiration period for the order equal 3 times the difference between the first two timestamps
    expiration_period = 3 * (studyStockData[1].timestamp - studyStockData[0].timestamp)
    print ("Expiration period: ", expiration_period)
    
    # Get the sATR value and the corresponding candle
    # def get_satr_value(candle):
    #     print ("Get sATR value")
    #     satr_record = StudyStockDataIndicatorValue.objects.filter(stockDataItem=candle, studyIndicator__indicator__name='sATR').first()
    #     if satr_record and satr_record.studyIndicator.indicator.name == 'sATR':
    #         value = json.loads(satr_record.value)['value']
    #         if math.isnan(value):
    #             return 0.00
    #         return float(value)
    #     return 0.00
    def get_satr_value(candle):
        """Return sATR value (float) for this candle using Study.mainSatrIndicator, with debug output."""
        # print(f"[get_satr_value] Candle id={candle.id}, ts={candle.timestamp}, study_id={candle.study_id}")

        # Verify study and mainSatrIndicator exist
        if not getattr(study, "mainSatrIndicator_id", None):
            # print("[get_satr_value] ❌ Study.mainSatrIndicator not set")
            return 0.0

        # print(f"[get_satr_value] Using StudyIndicator id={study.mainSatrIndicator_id}")

        # Query for sATR record
        rec = (
            StudyStockDataIndicatorValue.objects
            .filter(stockDataItem=candle, studyIndicator_id=study.mainSatrIndicator_id)
            .only("value")
            .first()
        )

        if not rec:
            # print("[get_satr_value] ⚠️ No StudyStockDataIndicatorValue found for this candle")
            return 0.0

        try:
            val_dict = json.loads(rec.value)
            raw_val = val_dict.get("value")
            # print(f"[get_satr_value] Raw JSON value: {raw_val}")

            if raw_val is None:
                # print("[get_satr_value] ⚠️ No 'value' key in JSON")
                return 0.0

            val = float(raw_val)
            if math.isnan(val):
                # print("[get_satr_value] ⚠️ NaN detected, returning 0.0")
                return 0.0

            # print(f"[get_satr_value] ✅ Parsed sATR value: {val}")
            return val

        except Exception as e:
            # print(f"[get_satr_value] ❌ Exception while parsing value: {e}")
            return 0.0


    # Get the corresponding candle for the indicator value (HOCLV)
    def get_candle(indicator_value):
        print("Get candle")
        return indicator_value.stockDataItem
    
    # Get the next candle for the current candle
    def get_next_candle(candle):
        try:
            return StockData.objects.filter(study=candle.study, timestamp__gt=candle.timestamp).order_by('timestamp').first()
        except StockData.DoesNotExist:
            return None

    def check_order_status(order, current_candle, expiration_period):
        # Check if the order's limit price has been reached
        if order.status == 'OPEN' and order.limitPrice >= current_candle.low and order.limitPrice <= current_candle.high:
            order.status = 'FILLED'
            order.filledAt=datetime.fromtimestamp(current_candle.timestamp / 1000)
            print("Order filled")

        # If the order is filled, check if the stop loss or take profit has been hit
        if order.status == 'FILLED':
            if order.stopLoss >= current_candle.low and order.stopLoss <= current_candle.high:
                order.status = 'CLOSED_BY_SL'
                order.closedAt=datetime.fromtimestamp(current_candle.timestamp / 1000)
                print("Order CLOSED_BY_SL")
            elif order.takeProfit >= current_candle.low and order.takeProfit <= current_candle.high:
                order.status = 'CLOSED_BY_TP'
                order.closedAt=datetime.fromtimestamp(current_candle.timestamp / 1000)
                print("Order CLOSED_BY_TP")

        # If the order is still open, check if it has expired
        if order.status == 'OPEN':
            if (current_candle.timestamp - order.stockDataItem.timestamp) >= expiration_period:
                order.status = 'EXPIRED'
                order.expiredAt=datetime.fromtimestamp(current_candle.timestamp / 1000)
                print("Order EXPIRED")

        # Save the order and remove it from the list if it is closed or expired
        if order.status in ['CLOSED_BY_SL', 'CLOSED_BY_TP', 'EXPIRED']:
            order.save()
            study_orders.remove(order)

        return order

    def generate_order(candle, next_candle, lpoffset, sl, tp, satr, direction):
        print("Function Generate order")
        # Calculate the order price, stop loss, and take profit based on the direction
        if direction == 'BUY':
            order_price = next_candle.open - lpoffset * satr
            stop_loss = order_price - sl * satr
            take_profit = order_price + tp * satr
        elif direction == 'SELL':
            order_price = next_candle.open + lpoffset * satr
            stop_loss = order_price + sl * satr
            take_profit = order_price - tp * satr

        # Create a StudyOrder
        order = StudyOrder(
            study=candle.study,
            stockDataItem=candle,
            orderType='LIMIT',
            quantity=1,  # Adjust this as needed
            limitPrice=order_price,
            takeProfit=take_profit,
            stopLoss=stop_loss,
            direction=direction,
            timeInForce='GTC',
            status='OPEN',
            createdAt=datetime.fromtimestamp(candle.timestamp / 1000),
            lpoffsetTP=lpoffset,
            slTP=sl,
            tpTP=tp,
        )

        return order

    study_orders = []
    i = 0
    for candle in studyStockData:
        i+=1
        print("Iteration No: ", i)

        # Get the sATR value for the current candle
        satr = get_satr_value(candle)
        print("sATR value: ", satr)
        if satr == 0.00:
            continue

        next_candle = get_next_candle(candle)
        if next_candle is None:
            continue      

        # Check StudyOrders for SL, TP, and expiration period
        for order in study_orders:
            order = check_order_status(order, candle, expiration_period)

        for lpoffset in tradingPlanParams['LPoffset']:
            for sl in tradingPlanParams['stopLoss']:
                for tp in tradingPlanParams['takeProfit']:
                    # Generate a StudyOrder
                    orderBuy = generate_order(candle, next_candle, lpoffset, sl, tp, satr, 'BUY')
                    orderSell = generate_order(candle, next_candle, lpoffset, sl, tp, satr, 'SELL')
                    study_orders.append(orderBuy)
                    study_orders.append(orderSell)
        
    # print("Indicators values: ", indicator_value.stockDataItem.open)  
    # print("Indicators values: ", indicator_value.studyIndicator.indicator.name)
    # print("Indicators values: ", indicator_value.value)
    # print("Trading plan params: ", tradingPlanParams)
    # print("Stock data: ", stock_data)

    # for candle in stock_data:
    #     print("open: ", candle.open)

    # Get the functionName for each StudyIndicator
    
    return "Trades simulated based on study indicators"