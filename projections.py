import pandas as pd
import warnings
from datetime import timedelta
import statsmodels


# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")

def create_advanced_sarimax_forecast(csv_file_path, forecast_days=14, festival_calendar=None):
    """
    Loads demand data, incorporates a festival calendar as an exogenous variable,
    fits a SARIMAX model, and returns a forecast with a correct DatetimeIndex.
    """
    if festival_calendar is None:
        festival_calendar = {}
        
    try:
        # Step 1: Load and Clean Data
        df = pd.read_csv(csv_file_path)
        df.columns = df.columns.str.strip()
        
        df_cleaned = df.dropna().copy()
        # Add a fixed year as the CSV doesn't contain it
        df_cleaned['DATE'] = pd.to_datetime('2024-' + df_cleaned['DATE'], format='%Y-%d-%b', errors='coerce')
        df_cleaned = df_cleaned.dropna(subset=['DATE'])
        df_cleaned.set_index('DATE', inplace=True)
        df_cleaned.sort_index(inplace=True)
        
        demand_series = df_cleaned['DEMAND']

        # Step 2: Create Exogenous Variable (dummy variable) for Festivals
        df_cleaned['is_festival'] = 0
        for date in festival_calendar.keys():
            if pd.Timestamp(date) in df_cleaned.index:
                df_cleaned.loc[pd.Timestamp(date), 'is_festival'] = 1
        
        exog_in_sample = df_cleaned[['is_festival']]

        # Step 3: Define and Fit the SARIMAX Model
        model = SARIMAX(demand_series,
                        exog=exog_in_sample,
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 7))
        results = model.fit(disp=False)

        # Step 4: Forecast with Future Exogenous Variables
        forecast_index = pd.date_range(start=demand_series.index[-1] + timedelta(days=1), periods=forecast_days)
        exog_forecast = pd.DataFrame(index=forecast_index)
        exog_forecast['is_festival'] = 0
        for date in festival_calendar.keys():
             if pd.Timestamp(date) in exog_forecast.index:
                exog_forecast.loc[pd.Timestamp(date), 'is_festival'] = 1

        forecast_object = results.get_forecast(steps=forecast_days, exog=exog_forecast)
        forecast_df = forecast_object.summary_frame(alpha=0.32) # ~68% confidence interval
        
        # ***** BUG FIX *****
        # Set the correct DatetimeIndex on the results DataFrame
        forecast_df.index = forecast_index
        
        # Apply the boost multiplier to the forecasted festival day
        for date, boost in festival_calendar.items():
            if pd.Timestamp(date) in forecast_df.index:
                forecast_df.loc[pd.Timestamp(date), 'mean'] *= boost

        return forecast_df

    except Exception as e:
        # Provide a specific error message if something goes wrong
        print(f"An error occurred in the forecasting function: {e}")
        return None


def generate_ramp_up_plan(forecast_df, current_stock_on_hand, festival_date, ramp_up_days=6, safety_stock_factor=0.2):
    """
    Generates a production ramp-up plan for a specific festival.
    """
    try:
        festival_date_ts = pd.Timestamp(festival_date)
        
        # Check if the festival date is within the forecast period
        if festival_date_ts not in forecast_df.index:
            return f"Error: Festival date {festival_date} is not within the forecast range."
            
        # Step 1: Calculate Target Inventory
        target_demand = forecast_df.loc[festival_date_ts, 'mean']
        target_inventory = target_demand * (1 + safety_stock_factor)

        # Step 2: Calculate Demand Until the Festival
        demand_until_festival = forecast_df.loc[forecast_df.index < festival_date_ts, 'mean'].sum()

        # Step 3: Calculate Total Production Need
        total_production_needed = (target_inventory + demand_until_festival) - current_stock_on_hand
        
        if total_production_needed <= 0:
            return "No production needed. Current stock is sufficient to meet demand."

        # Step 4: Create the Daily Ramp-Up Plan
        daily_production_target = total_production_needed / ramp_up_days
        
        plan = {
            "FESTIVAL ALERT": f"Major demand spike on {festival_date_ts.strftime('%Y-%m-%d')}",
            "Target Inventory Required": f"{target_inventory:.0f} units",
            "Total Production Needed": f"{total_production_needed:.0f} units",
            "Ramp-Up Period": f"{ramp_up_days} days",
            "Recommendation": "Begin immediate ramp-up. Suggested daily production:",
            "Daily Plan": {}
        }
        
        # Calculate the start date for the production plan
        plan_start_date = festival_date_ts - timedelta(days=ramp_up_days)
        
        for i in range(ramp_up_days):
            plan_date = (plan_start_date + timedelta(days=i)).strftime('%Y-%m-%d')
            plan["Daily Plan"][plan_date] = f"Produce {daily_production_target:.0f} units"
            
        return plan

    except Exception as e:
        # Provide a specific error message if something goes wrong
        return f"An error occurred in the planning function: {e}"


# --- Main execution block to run the complete system ---
if __name__ == "__main__":
    
    # --- Define Inputs (These would be dynamic in a live system) ---
    file_path = "400 G SELECT CUP-Table 1.csv"
    
    # The festival date MUST be within the forecast window.
    # The data ends 2024-06-15. A 14-day forecast runs until 2024-06-29.
    FESTIVAL_DATE = '2024-06-22' 
    FESTIVAL_CALENDAR = {FESTIVAL_DATE: 6.0} # A 6x demand multiplier for the festival
    
    CURRENT_SOH = 5000  # Your current stock on hand for the product category
    RAMP_UP_DAYS = 6    # How many days you have to build up stock
    FORECAST_HORIZON = 14 # How many days into the future to forecast

    print("--- Starting Forecasting and Planning ---")
    
    # --- Step 1: Generate the Forecast ---
    forecast = create_advanced_sarimax_forecast(
        file_path, 
        forecast_days=FORECAST_HORIZON, 
        festival_calendar=FESTIVAL_CALENDAR
    )
    
    if forecast is not None:
        print("\nSuccessfully generated SARIMAX forecast.")
        
        # --- Step 2: Generate the Production Ramp-Up Plan ---
        production_plan = generate_ramp_up_plan(
            forecast_df=forecast,
            current_stock_on_hand=CURRENT_SOH,
            festival_date=FESTIVAL_DATE,
            ramp_up_days=RAMP_UP_DAYS
        )

        print("\n--- Production Ramp-Up Plan ---")
        if isinstance(production_plan, dict):
            for key, value in production_plan.items():
                if isinstance(value, dict):
                    print(f"\n{key}:")
                    for sub_key, sub_value in value.items():
                        print(f"  - {sub_key}: {sub_value}")
                else:
                    print(f"- {key}: {value}")
        else:
            print(production_plan)
        print("---------------------------------")
    else:
        print("Could not generate a forecast. Halting process.")