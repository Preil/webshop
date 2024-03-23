# services.py
from .models import Study, StockData, StudyIndicator, StudyStockDataIndicatorValue
from django.core.exceptions import ObjectDoesNotExist
from . import calculations

INDICATOR_FUNCTIONS = {
    '#movingAverage': calculations.movingAverage,
    '#atr': calculations.atr,
    '#rsi': calculations.rsi,
    '#satr': calculations.satr,
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
    
# Simulate trades with trading plan
def simulate_trades(study,studyTradingPlan):
    # Retrieve the StockData for the study
    stock_data = StockData.objects.filter(study=study)

    # Retrieve the StudyIndicators for the study
    study_indicators = StudyIndicator.objects.filter(study=study)
    
    # Get the functionName for each StudyIndicator
    
    return "Trades simulated based on study indicators"