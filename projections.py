import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
import config
from utils.data_models import SKU
from utils.data_loader import DataLoader 
from typing import *
import math
import pickle
import plotly.graph_objects as go

FORECAST_HORIZON_DAYS = 9
MIN_SALES_HISTORY_DAYS_FOR_FORECAST = 30
dl = DataLoader()
config.SKUS.clear()
config.SKUS.update(dl.load_skus_with_fallback())
SKU_TO_CATEGORY_MAP = {sku_id: sku_obj.product_category for sku_id, sku_obj in config.SKUS.items()}

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads wide-format sales data, cleans it, normalizes sales,
    and returns a single, tidy DataFrame.
    """
    print(f"\nLoading and cleaning data from '{file_path}'...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: {file_path} not found.")

    df_wide = pd.read_csv(file_path)
    
    # Standardize Date column
    df_wide['Date'] = pd.to_datetime(df_wide['Date'], format='%d/%m/%y')

    # Identify SKU columns
    sku_sales_cols = [col for col in df_wide.columns if col != 'Date' and not col.endswith('_multiplier')]
    
    # Melt sales data into long format
    df_long = df_wide.melt(id_vars=['Date'], value_vars=sku_sales_cols, var_name='SKU', value_name='Sales')
    
    # Melt multiplier data separately
    sku_multiplier_cols = [f'{sku}_multiplier' for sku in sku_sales_cols]
    df_multipliers = df_wide.melt(id_vars=['Date'], value_vars=sku_multiplier_cols, var_name='SKU_Multiplier_Col', value_name='Multiplier')
    
    # Clean up multiplier SKU names and merge
    df_multipliers['SKU'] = df_multipliers['SKU_Multiplier_Col'].str.replace('_multiplier', '')
    df_long = pd.merge(df_long, df_multipliers[['Date', 'SKU', 'Multiplier']], on=['Date', 'SKU'])
    
    # Handle missing data before normalization
    df_long['Multiplier'] = df_long['Multiplier'].fillna(1.0)
    
    # Calculate Normalized Sales (the true underlying demand)
    # We add a small epsilon to avoid division by zero, though our data generator avoids zeros.
    df_long['Normalized_Sales'] = df_long['Sales'] / (df_long['Multiplier'] + 1e-6)
    
    # Set Date as the index for time series operations
    df_long = df_long.set_index('Date').sort_index()

    print("Data loading, cleaning, and normalization complete.")
    return df_long

def forecast_sku_demand(
    full_cleaned_data: pd.DataFrame, 
    current_simulation_date: pd.Timestamp
) -> dict:
    """
    Generates point demand forecasts for each SKU using dynamic history.
    """
    print(f"\nGenerating {FORECAST_HORIZON_DAYS}-day forecast as of {current_simulation_date.strftime('%Y-%m-%d')}...")
    
    forecasts = {}
    unique_skus = full_cleaned_data['SKU'].unique()
    forecast_dates = pd.date_range(start=current_simulation_date, periods=FORECAST_HORIZON_DAYS, freq='D')
    training_data = full_cleaned_data[full_cleaned_data.index < current_simulation_date]

    for sku in unique_skus:
        sku_series = training_data[training_data['SKU'] == sku]['Normalized_Sales'].asfreq('D')

        if pd.isna(sku_series[sku_series > 0].index.min()) or (current_simulation_date - sku_series[sku_series > 0].index.min()).days < MIN_SALES_HISTORY_DAYS_FOR_FORECAST:
            # Using zero forecast for SKUs with insufficient history
            forecasts[sku] = pd.Series([0.0] * FORECAST_HORIZON_DAYS, index=forecast_dates)
            continue

        if len(sku_series.dropna()) < 14:
            # Using simple average for SKUs with less than 14 data points
            avg_forecast = sku_series.mean()
            forecasts[sku] = pd.Series([avg_forecast] * FORECAST_HORIZON_DAYS, index=forecast_dates)
            continue

        try:
            # --- FIX: Suppress the specific ConvergenceWarning ---
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model = ExponentialSmoothing(
                    sku_series,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=4,
                    initialization_method="estimated"
                ).fit(
                    
                )

            
            sku_forecast = model.forecast(steps=FORECAST_HORIZON_DAYS)
            forecasts[sku] = sku_forecast.apply(lambda x: max(0, x))

        except Exception as e:
            print(f"  - ERROR: Model failed for {sku} ({e}). Forecasting with simple average.")
            avg_forecast = sku_series.mean()
            forecasts[sku] = pd.Series([avg_forecast] * FORECAST_HORIZON_DAYS, index=forecast_dates)

    print("SKU forecasting complete.")
    return forecasts

def calculate_empirical_prediction_intervals(sku_forecasts: dict) -> pd.DataFrame:
    """
    Converts a dictionary of SKU forecasts into a DataFrame and adds
    empirical prediction intervals (lower and upper bounds).
    
    Returns:
        pd.DataFrame: A DataFrame with columns for point, lower, and upper forecasts for each SKU.
    """
    print("\nCalculating empirical prediction intervals...")
    if not sku_forecasts:
        return pd.DataFrame()

    # Define the increasing deviation percentages for the forecast horizon
    deviation_map = {1: 0.25, 2: 0.30, 3: 0.30, 4: 0.35, 5: 0.35, 6: 0.40, 7: 0.45, 8: 0.45, 9: 0.50}
    
    all_forecasts_df = pd.DataFrame(sku_forecasts)
    
    # Create new columns for point, lower, and upper bounds
    for sku in all_forecasts_df.columns:
        # The point forecast is the original forecast
        all_forecasts_df[f'{sku}_point'] = all_forecasts_df[sku]
        
        # Calculate daily deviation and apply it for lower/upper bounds
        day_offset = (all_forecasts_df.index - all_forecasts_df.index.min()).days + 1
        deviation_factor = day_offset.map(deviation_map).fillna(0.40)
        
        all_forecasts_df[f'{sku}_lower'] = all_forecasts_df[f'{sku}_point'] * (1 - deviation_factor)
        all_forecasts_df[f'{sku}_upper'] = all_forecasts_df[f'{sku}_point'] * (1 + deviation_factor)
        
        # Ensure bounds are not negative
        all_forecasts_df[f'{sku}_lower'] = all_forecasts_df[f'{sku}_lower'].apply(lambda x: max(0, x))

    # Drop the original single-value forecast columns
    final_df = all_forecasts_df.drop(columns=list(sku_forecasts.keys()))
    
    print("Prediction intervals calculated.")
    return final_df

def apply_future_multipliers(
    normalized_forecast_df: pd.DataFrame, 
    full_historical_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Applies future multipliers to the normalized forecast to get the final sales forecast.
    """
    print("Applying future multipliers to forecast...")
    adjusted_forecast_df = normalized_forecast_df.copy()

    for sku_col_base in [c.replace('_point', '') for c in adjusted_forecast_df.columns if c.endswith('_point')]:
        # Find the multipliers for this specific SKU for the forecast dates
        sku_multipliers = full_historical_data[
            (full_historical_data['SKU'] == sku_col_base) &
            (full_historical_data.index.isin(adjusted_forecast_df.index))
        ]['Multiplier']

        if not sku_multipliers.empty:
            # Multiply point, lower, and upper bounds by the multiplier for each corresponding date
            adjusted_forecast_df[f'{sku_col_base}_point'] *= sku_multipliers
            adjusted_forecast_df[f'{sku_col_base}_lower'] *= sku_multipliers
            adjusted_forecast_df[f'{sku_col_base}_upper'] *= sku_multipliers

    # Fill any potential NaN values that resulted from the merge with the original value (for days with no multiplier)
    adjusted_forecast_df.fillna(normalized_forecast_df, inplace=True)
    
    return adjusted_forecast_df

def reconcile_forecasts(forecast_df: pd.DataFrame, sku_to_category_map: dict) -> pd.DataFrame:
    """
    Performs bottom-up reconciliation for forecasts AND target stock levels.
    """
    print("Reconciling forecasts and targets (Bottom-Up)...")
    reconciled_df = forecast_df.copy()
    unique_categories = sorted(list(set(sku_to_category_map.values())))

    # --- Reconcile Categories ---
    for category in unique_categories:
        skus_in_category = [sku for sku, cat in sku_to_category_map.items() if cat == category]
        
        # Define columns for forecasts
        point_cols = [f'{sku}_point' for sku in skus_in_category if f'{sku}_point' in reconciled_df.columns]
        lower_cols = [f'{sku}_lower' for sku in skus_in_category if f'{sku}_lower' in reconciled_df.columns]
        upper_cols = [f'{sku}_upper' for sku in skus_in_category if f'{sku}_upper' in reconciled_df.columns]
        # Define columns for target stock
        target_cols = [sku for sku in skus_in_category if sku in reconciled_df.columns]

        # Sum the forecasts
        if point_cols:
            reconciled_df[f'{category}_point'] = reconciled_df[point_cols].sum(axis=1)
            reconciled_df[f'{category}_lower'] = reconciled_df[lower_cols].sum(axis=1)
            reconciled_df[f'{category}_upper'] = reconciled_df[upper_cols].sum(axis=1)
        # Sum the target stocks
        if target_cols:
            reconciled_df[category] = reconciled_df[target_cols].sum(axis=1)
            
    # --- Reconcile Overall Total for both forecasts and targets ---
    cat_point_cols = [f'{cat}_point' for cat in unique_categories if f'{cat}_point' in reconciled_df.columns]
    if cat_point_cols:
        # Define category columns for lower, upper, and target
        cat_lower_cols = [f'{cat}_lower' for cat in unique_categories if f'{cat}_lower' in reconciled_df.columns]
        cat_upper_cols = [f'{cat}_upper' for cat in unique_categories if f'{cat}_upper' in reconciled_df.columns]
        cat_target_cols = [cat for cat in unique_categories if cat in reconciled_df.columns]
        
        # Sum all category columns to get the grand total
        reconciled_df['Total_Overall_point'] = reconciled_df[cat_point_cols].sum(axis=1)
        reconciled_df['Total_Overall_lower'] = reconciled_df[cat_lower_cols].sum(axis=1)
        reconciled_df['Total_Overall_upper'] = reconciled_df[cat_upper_cols].sum(axis=1)
        
        # Sum the category target stocks to get the overall target stock
        if cat_target_cols:
            reconciled_df['Total_Overall'] = reconciled_df[cat_target_cols].sum(axis=1)

    return reconciled_df

def calculate_target_stock_on_hand(
    forecast_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    current_date: pd.Timestamp,
    sku_list: list,
    sku_to_category_map: dict
) -> pd.DataFrame:
    """
    Calculates a target stock, now including proactive "build-ahead" logic
    for upcoming high-demand spikes identified by multipliers.
    """
    print(f"\nCalculating Proactive Daily Target Stock for {current_date.strftime('%Y-%m-%d')}...")

    # --- Configuration ---
    NORMAL_TARGET_FACTOR, MEDIUM_SPIKE_FACTOR, HEAVY_SPIKE_FACTOR = 1.75, 1.5, 1.2
    MEDIUM_SPIKE_THRESHOLD, HEAVY_SPIKE_THRESHOLD = 2.0, 3.5
    HIGH_VOLUME_THRESHOLD, HIGH_VOLUME_FACTOR = 10000, 1.2
    
    # --- NEW: Build-Ahead Logic Configuration ---
    BUILD_AHEAD_DAYS = 4  # How many days in advance to start building for a spike
    SPIKE_MULTIPLIER_THRESHOLD = 1.5 # What multiplier counts as a spike
    BUILD_AHEAD_FACTOR = 3.0 # Use a higher target factor to build stock

    # --- Main Logic ---
    daily_target_df = pd.DataFrame(index=forecast_df.index)
    recent_avg_sales = historical_df[historical_df.index < current_date].loc[current_date - timedelta(days=16):current_date - timedelta(days=2)].groupby('SKU')['Sales'].mean()

    for sku in sku_list:
        if f'{sku}_point' not in forecast_df.columns: continue

        point_forecasts = forecast_df[f'{sku}_point']
        sku_recent_avg = recent_avg_sales.get(sku, 0)
        daily_targets = []
        
        category = sku_to_category_map.get(sku)
        if not category: continue
        category_forecasts = forecast_df[f'{category}_point']

        for i in range(len(point_forecasts)):
            forecast_day = point_forecasts.index[i]
            
            # --- NEW: Check for upcoming spikes to enable build-ahead ---
            lookahead_end_date = forecast_day + timedelta(days=BUILD_AHEAD_DAYS)
            future_multipliers = historical_df[
                (historical_df['SKU'] == sku) &
                (historical_df.index > forecast_day) &
                (historical_df.index <= lookahead_end_date)
            ]
            is_spike_upcoming = any(future_multipliers['Multiplier'] > SPIKE_MULTIPLIER_THRESHOLD)
            
            sku_future_slice = point_forecasts.iloc[i+1 : i+3]
            category_future_slice = category_forecasts.iloc[i+1 : i+3]

            if len(sku_future_slice) > 0 and len(category_future_slice) > 0:
                avg_sku_demand = sku_future_slice.mean()
                avg_category_demand = category_future_slice.mean()

                # --- Dynamic Factor Logic now includes Build-Ahead ---
                if is_spike_upcoming:
                    target_factor = BUILD_AHEAD_FACTOR # Use special high factor
                elif avg_category_demand > HIGH_VOLUME_THRESHOLD:
                    target_factor = HIGH_VOLUME_FACTOR # Use category cap factor
                else:
                    # Proceed with the normal 3-tiered spike detection logic
                    target_factor = NORMAL_TARGET_FACTOR
                    if sku_recent_avg > 0:
                        demand_ratio = avg_sku_demand / sku_recent_avg
                        if demand_ratio > HEAVY_SPIKE_THRESHOLD: target_factor = HEAVY_SPIKE_FACTOR
                        elif demand_ratio > MEDIUM_SPIKE_THRESHOLD: target_factor = MEDIUM_SPIKE_FACTOR
                
                target = round((avg_sku_demand * target_factor) / 100) * 100
                daily_targets.append(target)
            else:
                if daily_targets: daily_targets.append(daily_targets[-1])
                else: daily_targets.append(0)

        daily_target_df[sku] = daily_targets

    return daily_target_df

def plot_rolling_forecast_performance(
    cleaned_df: pd.DataFrame, 
    all_rolling_forecasts: list, 
    sku_to_plot: str, 
    forecast_day_offset: int = 1
):
    """
    Plots the performance of a rolling forecast against actual sales for a specific SKU.

    Args:
        cleaned_df (pd.DataFrame): The full historical data.
        all_rolling_forecasts (list): A list of forecast DataFrames from the simulation.
        sku_to_plot (str): The SKU for which to plot the performance.
        forecast_day_offset (int): The forecast day to plot (e.g., 1 for D+1, 2 for D+2).
    """
    print(f"\nPlotting D+{forecast_day_offset} rolling forecast performance for {sku_to_plot}...")

    # Isolate actual sales for the SKU to plot
    actual_sales = cleaned_df[cleaned_df['SKU'] == sku_to_plot]['Sales'].dropna()
    
    # Extract the specific day's forecast from each simulation run
    point_col = f'{sku_to_plot}_point'
    rolling_predictions = []
    for forecast_df in all_rolling_forecasts:
        # Check if the forecast DataFrame is long enough and has the required column
        if len(forecast_df) >= forecast_day_offset and point_col in forecast_df.columns:
            # Get the Nth day's forecast
            nth_day_forecast = forecast_df.iloc[forecast_day_offset - 1]
            rolling_predictions.append(nth_day_forecast)

    if not rolling_predictions:
        print("Could not extract any rolling predictions to plot.")
        return
        
    predictions_df = pd.DataFrame(rolling_predictions).sort_index()

    # Plotting
    plt.figure(figsize=(18, 7))
    plt.plot(actual_sales.index, actual_sales.values, label='Actual Sales', color='blue', linewidth=2)
    plt.plot(predictions_df.index, predictions_df[point_col], label=f'D+{forecast_day_offset} Forecast', color='red', linestyle='--', alpha=0.8)
    
    plt.title(f'Rolling Forecast Performance vs. Actuals for {sku_to_plot}')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    filename = f"rolling_forecast_performance_{sku_to_plot}.png"
    plt.savefig(filename)
    print(f"Plot saved as '{filename}'")
    plt.show()

def plot_sku_summary(
    cleaned_df: pd.DataFrame,
    all_rolling_forecasts: list,
    sku_to_plot: str
):
    """
    Creates a comprehensive plot by extracting both forecast and target data
    from the main list of reconciled forecast DataFrames.
    """
    print(f"\nGenerating summary plot for {sku_to_plot}...")

    # --- Part 1: Prepare the data ---
    actual_sales = cleaned_df[cleaned_df['SKU'] == sku_to_plot]['Sales'].dropna()
    
    # Extract D+1 forecast and target data from each simulation run
    d1_data = []
    for forecast_df in all_rolling_forecasts:
        if f'{sku_to_plot}_point' in forecast_df.columns and not forecast_df.empty:
            d1_data.append(forecast_df.iloc[0]) # Get the entire first row (D+1)
            
    if not d1_data:
        print(f"  - No simulation data to plot for {sku_to_plot}.")
        return
    d1_data_df = pd.DataFrame(d1_data).sort_index()

    # Get the final, full 9-day forecast and targets
    final_full_forecast = all_rolling_forecasts[-1]
    
    # Define column names
    point_col, lower_col, upper_col = f'{sku_to_plot}_point', f'{sku_to_plot}_lower', f'{sku_to_plot}_upper'
    target_col = sku_to_plot # The target column is just the SKU name

    # --- Part 2: Create the plot ---
    plt.figure(figsize=(18, 7))
    
    plt.plot(actual_sales.index.to_numpy(), actual_sales.values, label='Actual Sales', color='blue')
    plt.plot(d1_data_df.index.to_numpy(), d1_data_df[point_col], label='Day-Ahead (D+1) Forecast', color='red', linestyle='--', alpha=0.7)
    plt.fill_between(d1_data_df.index.to_numpy(), d1_data_df[lower_col], d1_data_df[upper_col], color='red', alpha=0.15, label='Historical D+1 Interval')
    
    # Plot historical D+1 target stock line
    if target_col in d1_data_df.columns:
        plt.plot(d1_data_df.index.to_numpy(), d1_data_df[target_col], color='orange', linestyle='-.', linewidth=2, label='Historical D+1 Target')

    # Plot final forecast "cone" and its target line
    if point_col in final_full_forecast.columns:
        plt.plot(final_full_forecast.index.to_numpy(), final_full_forecast[point_col], color='purple', linewidth=2, label='Final 9-Day Forecast')
        plt.fill_between(final_full_forecast.index.to_numpy(), final_full_forecast[lower_col], final_full_forecast[upper_col], color='purple', alpha=0.2, label='Final Prediction Interval')
    if target_col in final_full_forecast.columns:
        plt.plot(final_full_forecast.index.to_numpy(), final_full_forecast[target_col], color='green', linestyle=':', linewidth=2, label='Final 9-Day Target Stock')

    plt.title(f'Sales & Forecast Summary for {sku_to_plot}')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    filename = f"summary_plot_{sku_to_plot}.png"
    plt.savefig(filename)
    print(f"  - Plot saved as '{filename}'")
    plt.close()

def plot_category_summary(
    cleaned_df: pd.DataFrame,
    all_rolling_forecasts: list,
    category_to_plot: str,
    sku_to_category_map: dict
):
    """
    Creates a summary plot for a reconciled product category.
    """
    print(f"\nGenerating summary plot for Category: {category_to_plot}...")

    # --- Part 1: Prepare the data ---
    
    # Find all SKUs that belong to this category
    skus_in_category = [sku for sku, cat in sku_to_category_map.items() if cat == category_to_plot]
    
    # Calculate the actual historical sales for the category by summing its SKUs
    actual_category_sales = cleaned_df[cleaned_df['SKU'].isin(skus_in_category)].groupby('Date')['Sales'].sum().dropna()

    # Extract D+1 reconciled forecast and target data
    d1_data = []
    for forecast_df in all_rolling_forecasts:
        if f'{category_to_plot}_point' in forecast_df.columns and not forecast_df.empty:
            d1_data.append(forecast_df.iloc[0])
            
    if not d1_data:
        print(f"  - No simulation data to plot for Category: {category_to_plot}.")
        return
    d1_data_df = pd.DataFrame(d1_data).sort_index()

    # Get the final, full 9-day forecast and targets
    final_full_forecast = all_rolling_forecasts[-1]
    
    # Define column names for the category
    point_col = f'{category_to_plot}_point'
    lower_col = f'{category_to_plot}_lower'
    upper_col = f'{category_to_plot}_upper'
    target_col = category_to_plot

    # --- Part 2: Create the plot ---
    plt.figure(figsize=(18, 7))
    
    plt.plot(actual_category_sales.index.to_numpy(), actual_category_sales.values, label='Actual Category Sales', color='blue')
    plt.plot(d1_data_df.index.to_numpy(), d1_data_df[point_col], label='D+1 Reconciled Forecast', color='red', linestyle='--', alpha=0.7)
    plt.fill_between(d1_data_df.index.to_numpy(), d1_data_df[lower_col], d1_data_df[upper_col], color='red', alpha=0.15, label='Historical D+1 Interval')
    
    if target_col in d1_data_df.columns:
        plt.plot(d1_data_df.index.to_numpy(), d1_data_df[target_col], color='orange', linestyle='-.', linewidth=2, label='Historical D+1 Target')

    if point_col in final_full_forecast.columns:
        plt.plot(final_full_forecast.index.to_numpy(), final_full_forecast[point_col], color='purple', linewidth=2, label='Final 9-Day Forecast')
        plt.fill_between(final_full_forecast.index.to_numpy(), final_full_forecast[lower_col], final_full_forecast[upper_col], color='purple', alpha=0.2, label='Final Prediction Interval')
    if target_col in final_full_forecast.columns:
        plt.plot(final_full_forecast.index.to_numpy(), final_full_forecast[target_col], color='green', linestyle=':', linewidth=2, label='Final 9-Day Target Stock')

    plt.title(f'Sales & Forecast Summary for Category: {category_to_plot}')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    filename = f"summary_plot_CATEGORY_{category_to_plot}.png"
    plt.savefig(filename)
    print(f"  - Plot saved as '{filename}'")
    plt.close()

def get_initial_inventory(
    forecast_df: pd.DataFrame,
    lead_time_days: int,
    coverage_days: int,
    sku_list: list  # <-- Add sku_list as an argument
) -> Tuple[dict, dict]:
    """
    Calculates a realistic starting inventory scenario for SKUs only.
    """
    print("\nSetting up initial inventory scenario...")
    initial_on_hand = {}
    initial_on_order = {}

    # UPDATED: We now loop through the explicit sku_list provided
    for sku in sku_list:
        # Check if the sku exists in the target stock columns
        if sku in forecast_df.columns:
            # 1. Set initial on-hand to the Day 1 target stock level
            on_hand = forecast_df[sku].iloc[0]
            initial_on_hand[sku] = on_hand

            # 2. Calculate a smart on-order quantity
            demand_during_lead_time = forecast_df[f'{sku}_point'].iloc[0:lead_time_days].sum()
            stock_at_arrival = on_hand - demand_during_lead_time
            demand_post_arrival = forecast_df[f'{sku}_point'].iloc[lead_time_days : lead_time_days + coverage_days].sum()

            deficit = max(0, demand_post_arrival - stock_at_arrival)
            initial_on_order[sku] = round(deficit / 100) * 100
        else:
            # If the SKU has no target stock, start with zero inventory
            initial_on_hand[sku] = 0
            initial_on_order[sku] = 0

    return initial_on_hand, initial_on_order

def project_inventory_forward(
    initial_on_hand: dict,
    production_pipeline: list, # <-- Accepts the full pipeline
    forecast_df: pd.DataFrame,
    sku_to_category_map: dict  # <-- Accepts the map to allocate arriving stock
) -> Tuple[pd.DataFrame, list]:
    """
    Projects inventory levels forward, accounting for a full pipeline of
    future production orders arriving on different dates.
    """
    print("\nProjecting future inventory levels with full production pipeline...")
    
    projected_inventory_df = pd.DataFrame(index=forecast_df.index)
    stockout_events = []
    items_to_project = list(initial_on_hand.keys())

    # Create a lookup for which orders arrive on which date for efficiency
    arrivals = {}
    for order in production_pipeline:
        arrival_date = pd.to_datetime(order['Arrival Date'])
        if arrival_date not in arrivals:
            arrivals[arrival_date] = []
        arrivals[arrival_date].append(order)

    # Project for each SKU
    for sku in items_to_project:
        current_stock = initial_on_hand.get(sku, 0)
        daily_projection = []
        
        for date, row in forecast_df.iterrows():
            # Check if any orders for this SKU's category arrive today
            if date in arrivals:
                for order in arrivals[date]:
                    # If this SKU is in the arriving order's category, allocate a share of the stock
                    if sku_to_category_map.get(sku) == order['Item']:
                        skus_in_cat = [s for s, c in sku_to_category_map.items() if c == order['Item']]
                        # Simple equal allocation of the batch to the SKUs in the category
                        if len(skus_in_cat) > 0:
                            current_stock += order['Quantity (L)'] / len(skus_in_cat)

            # Use the point forecast for demand, default to 0 if not found
            demand = row.get(f'{sku}_point', 0)
            
            # Identify potential stockouts *before* subtracting demand
            if current_stock < demand:
                stockout_events.append({'Date': date, 'Item': sku, 'Deficit': demand - current_stock})

            # Subtract the day's demand
            current_stock -= demand
            current_stock = max(0, current_stock) # Inventory can't go below zero
            daily_projection.append(current_stock)
        
        projected_inventory_df[sku] = daily_projection
        
    return projected_inventory_df, stockout_events

def generate_production_plan(
    projected_sku_inventory: pd.DataFrame,
    reconciled_forecast_df: pd.DataFrame,
    sku_to_category_map: dict,
    lead_time_days: int,
    batch_size: int
) -> pd.DataFrame:
    """
    Generates a production plan based on when a CATEGORY'S projected inventory
    is expected to drop below its target, calculating the required number of batches.
    """
    print("\nGenerating category-level production plan...")
    production_orders = []
    unique_categories = sorted(list(set(sku_to_category_map.values())))

    # Step 1: Aggregate projected SKU inventory to the Category level
    projected_category_inventory = pd.DataFrame(index=projected_sku_inventory.index)
    for category in unique_categories:
        skus_in_category = [sku for sku, cat in sku_to_category_map.items() if cat == category and sku in projected_sku_inventory.columns]
        if skus_in_category:
            projected_category_inventory[category] = projected_sku_inventory[skus_in_category].sum(axis=1)

    # Step 2: Determine production triggers for each category
    for category in unique_categories:
        if category not in projected_category_inventory.columns or category not in reconciled_forecast_df.columns:
            print(f"  - WARNING: No data for category '{category}'. Skipping production planning.")
            continue

        df = pd.concat([projected_category_inventory[category], reconciled_forecast_df[category]], axis=1)
        df.columns = ['Projected', 'Target']
        trigger_days = df.index[df['Projected'] < df['Target']]

        if not trigger_days.empty:
            first_trigger_date = trigger_days.min()
            
            # --- NEW LOGIC: Calculate deficit and number of batches ---
            projected_on_trigger = df.loc[first_trigger_date, 'Projected']
            target_on_trigger = df.loc[first_trigger_date, 'Target']
            deficit = target_on_trigger - projected_on_trigger
            
            if deficit > 0:
                # Calculate how many batches are needed to cover the deficit (rounding up)
                num_batches = math.ceil(deficit / batch_size)
                order_quantity = num_batches * batch_size
                
                order_date = first_trigger_date - timedelta(days=lead_time_days)
                
                production_orders.append({
                    "Item": category,
                    "Order Date": order_date.strftime('%Y-%m-%d'),
                    "Arrival Date": first_trigger_date.strftime('%Y-%m-%d'),
                    "Quantity (L)": order_quantity, # <-- Use the calculated quantity
                    "Reason": f"Stock projected {deficit:.0f}L below target on {first_trigger_date.strftime('%Y-%m-%d')}"
                })

    if not production_orders:
        return pd.DataFrame()

    return pd.DataFrame(production_orders)

def generate_sku_allocation_plan(
    production_plan: pd.DataFrame,
    projected_sku_inventory: pd.DataFrame,
    reconciled_forecast_df: pd.DataFrame,
    sku_to_category_map: dict
) -> pd.DataFrame:
    """
    Takes a category-level production plan and allocates the batch size
    down to the individual SKUs, with a hard cap and rounding.
    """
    print("\nAllocating batch production to individual SKUs...")
    sku_level_plan = []
    
    MAX_ALLOCATION_PER_CATEGORY = 10000
    ROUNDING_MULTIPLE = 50

    for _, order in production_plan.iterrows():
        category = order['Item']
        available_to_allocate = min(order['Quantity (L)'], MAX_ALLOCATION_PER_CATEGORY)
        arrival_date = pd.to_datetime(order['Arrival Date'])
        skus_in_category = [sku for sku, cat in sku_to_category_map.items() if cat == category]
        
        sku_deficits = {}
        total_category_deficit = 0

        for sku in skus_in_category:
            if sku not in reconciled_forecast_df.columns or sku not in projected_sku_inventory.columns:
                continue
            target_on_arrival = reconciled_forecast_df.loc[arrival_date, sku]
            projected_on_arrival = projected_sku_inventory.loc[arrival_date, sku]
            deficit = max(0, target_on_arrival - projected_on_arrival)
            sku_deficits[sku] = deficit
            total_category_deficit += deficit

        if total_category_deficit > 0:
            final_allocations = {}
            running_total = 0
            
            # Allocate proportionally and round, ensuring we don't exceed the cap
            for sku, deficit in sku_deficits.items():
                proportion = deficit / total_category_deficit
                allocated_quantity = available_to_allocate * proportion
                final_allocations[sku] = round(allocated_quantity / ROUNDING_MULTIPLE) * ROUNDING_MULTIPLE

            # --- NEW: Check and adjust the total after rounding ---
            total_after_rounding = sum(final_allocations.values())
            if total_after_rounding > available_to_allocate:
                # If rounding pushed the total over, trim the excess from the largest allocation
                excess = total_after_rounding - available_to_allocate
                sku_to_trim = max(final_allocations, key=final_allocations.get)
                final_allocations[sku_to_trim] -= excess
                # Ensure the trimmed value is still a multiple of 50
                final_allocations[sku_to_trim] = round(final_allocations[sku_to_trim] / ROUNDING_MULTIPLE) * ROUNDING_MULTIPLE

            for sku, final_quantity in final_allocations.items():
                if final_quantity > 0:
                    sku_level_plan.append({
                        "SKU": sku,
                        "Order Date": order['Order Date'],
                        "Arrival Date": order['Arrival Date'],
                        "Allocated Quantity (L)": final_quantity,
                        "Category": category
                    })

    if not sku_level_plan:
        return pd.DataFrame()

    return pd.DataFrame(sku_level_plan)

def plot_sku_production(
    sku_allocation_plan: pd.DataFrame,
    sku_to_plot: str
):
    """
    Creates a bar chart visualizing the production schedule for a single SKU.
    """
    print(f"\nGenerating SKU production plot for {sku_to_plot}...")
    
    df = sku_allocation_plan.copy()
    sku_df = df[df['SKU'] == sku_to_plot]

    if sku_df.empty:
        print(f"  - No production scheduled for SKU: {sku_to_plot}.")
        return

    sku_df['Order Date'] = pd.to_datetime(sku_df['Order Date'])
    
    plt.figure()
    plt.bar(sku_df['Order Date'], sku_df['Allocated Quantity (L)'], width=0.8, label=f'Production for {sku_to_plot}')
    
    plt.title(f'Production Schedule for SKU: {sku_to_plot}')
    plt.xlabel('Production Order Date')
    plt.ylabel('Quantity (L)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    filename = f"production_schedule_SKU_{sku_to_plot}.png"
    plt.savefig(filename)
    print(f"  - Plot saved as '{filename}'")
    plt.close()

def plot_category_production(
    sku_allocation_plan: pd.DataFrame,
    category_to_plot: str
):
    """
    Creates a stacked bar chart visualizing the production schedule for a product category.
    """
    print(f"\nGenerating Category production plot for {category_to_plot}...")

    df = sku_allocation_plan.copy()
    cat_df = df[df['Category'] == category_to_plot]

    if cat_df.empty:
        print(f"  - No production scheduled for Category: {category_to_plot}.")
        return

    cat_df['Order Date'] = pd.to_datetime(cat_df['Order Date'])
    pivot_df = cat_df.pivot_table(
        index='Order Date', 
        columns='SKU', 
        values='Allocated Quantity (L)',
        aggfunc='sum'
    ).fillna(0)

    plt.figure()
    pivot_df.plot(kind='bar', stacked=True, figsize=(18, 8), width=0.8)

    plt.title(f'Production Schedule for Category: {category_to_plot}')
    plt.xlabel('Production Order Date')
    plt.ylabel('Total Quantity (L)')
    plt.xticks(rotation=45)
    plt.legend(title='SKU', bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    filename = f"production_schedule_CATEGORY_{category_to_plot}.png"
    plt.savefig(filename)
    print(f"  - Plot saved as '{filename}'")
    plt.close()

def summarize_capacity_usage(production_plan_df: pd.DataFrame, capacity_groups: dict) -> pd.DataFrame:
    """
    Summarizes daily production for each capacity group, applying a dynamic
    capacity reduction based on the number of unique categories produced.
    """
    print("\nSummarizing daily capacity usage with dynamic reduction...")
    if production_plan_df.empty:
        return pd.DataFrame()

    # --- Configuration for Dynamic Capacity ---
    CAPACITY_REDUCTION_PER_CHANGEOVER = 0.075

    # Create a map from category to its capacity group and limit
    category_to_capacity_map = {}
    for group, details in capacity_groups.items():
        for category in details['categories']:
            category_to_capacity_map[category] = {'group': group, 'limit': details['daily_capacity']}

    plan_df = production_plan_df.copy()
    plan_df['Capacity Group'] = plan_df['Item'].map(lambda x: category_to_capacity_map.get(x, {}).get('group'))
    plan_df['Daily Capacity'] = plan_df['Item'].map(lambda x: category_to_capacity_map.get(x, {}).get('limit'))
    plan_df.dropna(subset=['Capacity Group'], inplace=True)

    # Grouping now also counts the number of unique categories to find changeovers
    summary = plan_df.groupby(['Order Date', 'Capacity Group', 'Daily Capacity']).agg(
        Total_Planned_L=('Quantity (L)', 'sum'),
        Categories_Produced=('Item', 'nunique') # Get count of unique items (categories)
    ).reset_index()

    # Calculate and apply the capacity reduction
    # The number of changeovers is one less than the number of categories produced
    summary['Num_Changeovers'] = summary['Categories_Produced'] - 1
    
    # Calculate the total reduction factor
    summary['Reduction_Factor'] = summary['Num_Changeovers'] * CAPACITY_REDUCTION_PER_CHANGEOVER
    
    # Calculate the adjusted capacity for the day
    summary['Adjusted_Capacity'] = summary['Daily Capacity'] * (1 - summary['Reduction_Factor'])

    # Flag any days where the plan exceeds the *adjusted* capacity
    summary['Alert'] = np.where(summary['Total_Planned_L'] > summary['Adjusted_Capacity'], 'CAPACITY EXCEEDED', 'OK')
    
    # Select and reorder columns for a clean final report
    final_report = summary[[
        'Order Date', 'Capacity Group', 'Daily Capacity', 'Adjusted_Capacity',
        'Total_Planned_L', 'Categories_Produced', 'Alert'
    ]]
    
    return final_report

def run_or_load_simulation(cache_path="simulation_cache.pkl", force_rerun=False, sim_end_date_override=None):
    """
    Orchestrates the simulation. Caching is bypassed if a specific end date is given.

    Args:
        cache_path (str): The path to the file where results are cached.
        force_rerun (bool): If True, the simulation will run even if a cache exists.
        sim_end_date_override (pd.Timestamp, optional): If provided, the simulation
                                                       will run to this specific date
                                                       and caching will be bypassed.

    Returns:
        tuple: A tuple containing the simulation results.
    """
    # Caching is only used for full runs (when force_rerun is False and no override date is given)
    if not force_rerun and sim_end_date_override is None and os.path.exists(cache_path):
        print(f"\n--- Loading simulation results from cache: '{cache_path}' ---")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    if sim_end_date_override:
        print(f"\n--- Starting new simulation until specified end date: {sim_end_date_override.strftime('%Y-%m-%d')} (Caching disabled) ---")
    else:
        print("\n--- No cache found or rerun forced. Starting new simulation. ---")

    # --- Configuration ---
    DATA_FILE = 'historical_sales_data_wide.csv'
    PRODUCTION_LEAD_TIME_DAYS = 2
    PRODUCTION_BATCH_SIZE = 5000

    # --- Step 1: Load Data ---
    cleaned_df = load_and_clean_data(DATA_FILE)

    # --- Step 2: Initialize the Simulation State ---
    print("\n--- Initializing Simulation State ---")
    simulation_start_date = cleaned_df.index.min() + timedelta(days=MIN_SALES_HISTORY_DAYS_FOR_FORECAST)
    
    # MODIFIED LINE: Use the override date if provided, otherwise use the last date from the data
    simulation_end_date = sim_end_date_override if sim_end_date_override else cleaned_df.dropna(subset=['Sales']).index.max()
    
    # Ensure the simulation period is valid
    if simulation_start_date > simulation_end_date:
        raise ValueError("Simulation start date cannot be after the end date.")

    initial_forecast_df = reconcile_forecasts(apply_future_multipliers(calculate_empirical_prediction_intervals(forecast_sku_demand(cleaned_df, simulation_start_date)), cleaned_df), SKU_TO_CATEGORY_MAP)
    initial_target_stock_df = calculate_target_stock_on_hand(initial_forecast_df, cleaned_df, simulation_start_date, list(config.SKUS.keys()), SKU_TO_CATEGORY_MAP)

    current_on_hand = {sku: initial_target_stock_df[sku].iloc[0] for sku in config.SKUS.keys() if sku in initial_target_stock_df.columns}

    production_pipeline = []
    sku_allocation_pipeline = []
    all_rolling_forecasts = []

    # --- Step 3: Run the Day-by-Day Simulation Loop ---
    print(f"\n--- Starting Full Inventory & Production Simulation ---\nPeriod: {simulation_start_date.strftime('%Y-%m-%d')} to {simulation_end_date.strftime('%Y-%m-%d')}")

    for sim_date in pd.date_range(start=simulation_start_date, end=simulation_end_date):
        print(f"\n----- Simulating Day: {sim_date.strftime('%Y-%m-%d')} -----")
        # A. Update current inventory based on yesterday's sales and arrivals
        yesterday = sim_date - timedelta(days=1)
        # Handle arrivals
        for order in production_pipeline:
            if pd.to_datetime(order['Arrival Date']) == sim_date:
                print(f"  - Production order arrived for {order['Item']} (+{order['Quantity (L)']})")
                skus_in_cat = [s for s, c in SKU_TO_CATEGORY_MAP.items() if c == order['Item']]
                if skus_in_cat:
                    for s in skus_in_cat:
                        if s in current_on_hand:
                            current_on_hand[s] += order['Quantity (L)'] / len(skus_in_cat)
        # Handle sales
        actual_sales_yesterday = cleaned_df[cleaned_df.index == yesterday]
        if not actual_sales_yesterday.empty:
            for _, row in actual_sales_yesterday.iterrows():
                sku, sales = row['SKU'], row['Sales']
                if sku in current_on_hand:
                    current_on_hand[sku] = max(0, current_on_hand[sku] - sales)

        # B. Run forecasting and reconciliation for the current day
        sku_point_forecasts = forecast_sku_demand(cleaned_df, current_simulation_date=sim_date)
        adjusted_sku_forecast_df = apply_future_multipliers(calculate_empirical_prediction_intervals(sku_point_forecasts), cleaned_df)
        reconciled_forecasts_only_df = reconcile_forecasts(adjusted_sku_forecast_df, SKU_TO_CATEGORY_MAP)
        sku_target_stock_df = calculate_target_stock_on_hand(reconciled_forecasts_only_df, cleaned_df, sim_date, list(config.SKUS.keys()), SKU_TO_CATEGORY_MAP)
        final_reconciled_df = reconcile_forecasts(pd.concat([adjusted_sku_forecast_df, sku_target_stock_df], axis=1), SKU_TO_CATEGORY_MAP)
        all_rolling_forecasts.append(final_reconciled_df)

        # C. Project inventory forward and plan production
        projected_inventory_df, _ = project_inventory_forward(current_on_hand, production_pipeline, final_reconciled_df, SKU_TO_CATEGORY_MAP)
        todays_production_plan = generate_production_plan(projected_inventory_df, final_reconciled_df, SKU_TO_CATEGORY_MAP, PRODUCTION_LEAD_TIME_DAYS, PRODUCTION_BATCH_SIZE)

        # D. Allocate SKUs if a new production order was generated
        if not todays_production_plan.empty:
            print("  - New production order generated!")
            production_pipeline.extend(todays_production_plan.to_dict('records'))
            todays_sku_allocation = generate_sku_allocation_plan(todays_production_plan, projected_inventory_df, final_reconciled_df, SKU_TO_CATEGORY_MAP)
            if not todays_sku_allocation.empty:
                sku_allocation_pipeline.extend(todays_sku_allocation.to_dict('records'))

    print("\n--- Simulation Complete ---")

    # --- Step 4: Final Outputs ---
    production_plan_df = pd.DataFrame(production_pipeline)
    from config import CAPACITY_GROUPS
    capacity_summary = summarize_capacity_usage(production_plan_df, CAPACITY_GROUPS)

    # Bundle results for returning
    results = (cleaned_df, all_rolling_forecasts, production_pipeline, sku_allocation_pipeline, capacity_summary)
    
    # Only save to cache if it was a full run (no date override)
    if sim_end_date_override is None:
        print(f"\n--- Saving simulation results to cache: '{cache_path}' ---")
        with open(cache_path, 'wb') as f:
            pickle.dump(results, f)

    return results

def generate_sku_summary_figure(cleaned_df: pd.DataFrame, all_rolling_forecasts: list, sku_to_plot: str) -> go.Figure:
    """Creates a comprehensive Plotly figure for a single SKU with a date range slider."""
    actual_sales = cleaned_df[cleaned_df['SKU'] == sku_to_plot]['Sales'].dropna()
    
    point_col = f'{sku_to_plot}_point'
    d1_data = [f.iloc[0] for f in all_rolling_forecasts if point_col in f.columns and not f.empty]
    
    if not d1_data:
        return go.Figure().update_layout(title=f"No Simulation Data for {sku_to_plot}")
        
    d1_data_df = pd.DataFrame(d1_data).sort_index()
    final_full_forecast = all_rolling_forecasts[-1]
    
    lower_col, upper_col = f'{sku_to_plot}_lower', f'{sku_to_plot}_upper'
    target_col = sku_to_plot

    fig = go.Figure()
    # --- All fig.add_trace calls remain the same ---
    fig.add_trace(go.Scatter(x=actual_sales.index, y=actual_sales.values, name='Actual Sales', mode='lines', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=d1_data_df.index, y=d1_data_df[point_col], name='Day-Ahead (D+1) Forecast', mode='lines', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=d1_data_df.index, y=d1_data_df[upper_col], name='D+1 Upper Bound', mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=d1_data_df.index, y=d1_data_df[lower_col], name='D+1 Interval', mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)'))
    if target_col in d1_data_df.columns:
        fig.add_trace(go.Scatter(x=d1_data_df.index, y=d1_data_df[target_col], name='Historical D+1 Target', mode='lines', line=dict(color='orange', dash='dashdot')))
    if point_col in final_full_forecast.columns:
        fig.add_trace(go.Scatter(x=final_full_forecast.index, y=final_full_forecast[point_col], name='Final 9-Day Forecast', mode='lines', line=dict(color='purple', width=3)))
        fig.add_trace(go.Scatter(x=final_full_forecast.index, y=final_full_forecast[upper_col], name='Final Upper Bound', mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=final_full_forecast.index, y=final_full_forecast[lower_col], name='Final Prediction Interval', mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(128, 0, 128, 0.2)'))
    if target_col in final_full_forecast.columns:
        fig.add_trace(go.Scatter(x=final_full_forecast.index, y=final_full_forecast[target_col], name='Final 9-Day Target Stock', mode='lines', line=dict(color='green', dash='dot')))

    # --- MODIFICATION START ---
    # Calculate the default 2-week window for the slider
    end_date = actual_sales.index.max()
    start_date = end_date - pd.to_timedelta('14 days')


    fig.update_layout(
        title=f'Sales & Forecast Summary for {sku_to_plot}',
        xaxis_title='Date',
        yaxis_title='Sales',
        legend_title='Metric',
        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
            type="date",
            range=[start_date, end_date] # Set the default view window
        )
    )

    return fig

def generate_category_summary_figure(cleaned_df: pd.DataFrame, all_rolling_forecasts: list, category_to_plot: str, sku_to_category_map: dict) -> go.Figure:
    """Creates a comprehensive Plotly figure for a category with a date range slider."""
    skus_in_category = [sku for sku, cat in sku_to_category_map.items() if cat == category_to_plot]
    actual_category_sales = cleaned_df[cleaned_df['SKU'].isin(skus_in_category)].groupby('Date')['Sales'].sum().dropna()

    point_col = f'{category_to_plot}_point'
    d1_data = [f.iloc[0] for f in all_rolling_forecasts if point_col in f.columns and not f.empty]
    
    if not d1_data:
        return go.Figure().update_layout(title=f"No Simulation Data for Category: {category_to_plot}")
        
    d1_data_df = pd.DataFrame(d1_data).sort_index()
    final_full_forecast = all_rolling_forecasts[-1]
    
    lower_col = f'{category_to_plot}_lower'
    upper_col = f'{category_to_plot}_upper'
    target_col = category_to_plot

    fig = go.Figure()
    # --- All fig.add_trace calls remain the same ---
    fig.add_trace(go.Scatter(x=actual_category_sales.index, y=actual_category_sales.values, name='Actual Category Sales', mode='lines', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=d1_data_df.index, y=d1_data_df[point_col], name='Day-Ahead (D+1) Forecast', mode='lines', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=d1_data_df.index, y=d1_data_df[upper_col], name='D+1 Upper Bound', mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=d1_data_df.index, y=d1_data_df[lower_col], name='D+1 Interval', mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)'))
    if target_col in d1_data_df.columns:
        fig.add_trace(go.Scatter(x=d1_data_df.index, y=d1_data_df[target_col], name='Historical D+1 Target', mode='lines', line=dict(color='orange', dash='dashdot')))
    if point_col in final_full_forecast.columns:
        fig.add_trace(go.Scatter(x=final_full_forecast.index, y=final_full_forecast[point_col], name='Final 9-Day Forecast', mode='lines', line=dict(color='purple', width=3)))
        fig.add_trace(go.Scatter(x=final_full_forecast.index, y=final_full_forecast[upper_col], name='Final Upper Bound', mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=final_full_forecast.index, y=final_full_forecast[lower_col], name='Final Prediction Interval', mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(128, 0, 128, 0.2)'))
    if target_col in final_full_forecast.columns:
        fig.add_trace(go.Scatter(x=final_full_forecast.index, y=final_full_forecast[target_col], name='Final 9-Day Target Stock', mode='lines', line=dict(color='green', dash='dot')))

    # --- MODIFICATION START ---
    # Calculate the default 2-week window for the slider
    end_date = actual_category_sales.index.max()
    start_date = end_date - pd.to_timedelta('14 days')

    # Update the layout to include a rangeslider
    fig.update_layout(
        title=f'Sales & Forecast Summary for Category: {category_to_plot}',
        xaxis_title='Date',
        yaxis_title='Sales',
        legend_title='Metric',
        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
            type="date",
            range=[start_date, end_date] # Set the default view window
        )
    )
    # --- MODIFICATION END ---
    
    return fig


def generate_sku_production_figure(sku_allocation_plan: pd.DataFrame, sku_to_plot: str) -> go.Figure:
    """Creates a Plotly bar chart visualizing the production schedule for a single SKU."""
    sku_df = sku_allocation_plan[sku_allocation_plan['SKU'] == sku_to_plot]
    if sku_df.empty:
        return go.Figure().update_layout(title=f"No Production Scheduled for SKU: {sku_to_plot}", showlegend=False)
        
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pd.to_datetime(sku_df['Order Date']),
        y=sku_df['Allocated Quantity (L)'],
        name=f'Production for {sku_to_plot}'
    ))
    fig.update_layout(
        title=f'Production Schedule for SKU: {sku_to_plot}',
        xaxis_title='Production Order Date',
        yaxis_title='Quantity (L)',
        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    return fig

def generate_category_production_figure(sku_allocation_plan: pd.DataFrame, category_to_plot: str) -> go.Figure:
    """Creates a Plotly stacked bar chart for a product category's production schedule."""
    cat_df = sku_allocation_plan[sku_allocation_plan['Category'] == category_to_plot]
    if cat_df.empty:
        return go.Figure().update_layout(title=f"No Production Scheduled for Category: {category_to_plot}")

    pivot_df = cat_df.pivot_table(
        index='Order Date', 
        columns='SKU', 
        values='Allocated Quantity (L)',
        aggfunc='sum'
    ).fillna(0)
    pivot_df.index = pd.to_datetime(pivot_df.index)

    fig = go.Figure()
    for sku in pivot_df.columns:
        fig.add_trace(go.Bar(
            x=pivot_df.index,
            y=pivot_df[sku],
            name=sku
        ))

    fig.update_layout(
        barmode='stack',
        title=f'Production Schedule for Category: {category_to_plot}',
        xaxis_title='Production Order Date',
        yaxis_title='Total Quantity (L)',
        legend_title='SKU',
        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    return fig

def get_projections_and_plots(projection_date: str = None, force_rerun: bool = False) -> dict:
    """
    Main function to run the simulation and generate all data and Plotly figures.
    This version returns a structured, nested dictionary for plots.
    """
    # --- Step 1: Run Simulation or Load from Cache ---
    use_cache = not force_rerun and projection_date is None
    sim_end_date = pd.to_datetime(projection_date) if projection_date else None
    
    cleaned_df, all_rolling_forecasts, production_pipeline, sku_allocation_pipeline, capacity_summary = \
        run_or_load_simulation(force_rerun=not use_cache, sim_end_date_override=sim_end_date)

    # --- Step 2: Prepare Data and Plot Dictionaries ---
    
    # CORRECT: Initialize a nested dictionary structure for the plots
    plot_figures = {
        "sku_summary_figs": {},
        "category_summary_figs": {},
        "sku_production_figs": {},
        "category_production_figs": {}
    }

    if all_rolling_forecasts:
        unique_skus = cleaned_df['SKU'].unique()
        unique_categories = sorted(list(set(SKU_TO_CATEGORY_MAP.values())))

        print("\n--- Generating all Plotly figures for application ---")

        # CORRECT: Populate the nested dictionaries
        for sku in unique_skus:
            plot_figures["sku_summary_figs"][sku] = generate_sku_summary_figure(cleaned_df, all_rolling_forecasts, sku)
        for cat in unique_categories:
            plot_figures["category_summary_figs"][cat] = generate_category_summary_figure(cleaned_df, all_rolling_forecasts, cat, SKU_TO_CATEGORY_MAP)
            
        sku_allocation_plan_df = pd.DataFrame(sku_allocation_pipeline)
        if not sku_allocation_plan_df.empty:
            for sku in unique_skus:
                plot_figures["sku_production_figs"][sku] = generate_sku_production_figure(sku_allocation_plan_df, sku)
            for cat in unique_categories:
                plot_figures["category_production_figs"][cat] = generate_category_production_figure(sku_allocation_plan_df, cat)
    
    print("\n--- Projection and plot generation complete. ---")

    # CORRECT: Combine data and the nested plot dictionary for the final output
    final_output = {
        "production_plan": pd.DataFrame(production_pipeline),
        "sku_allocation_plan": pd.DataFrame(sku_allocation_pipeline),
        "capacity_summary": capacity_summary,
        "full_cleaned_data": cleaned_df,
        "all_rolling_forecasts": all_rolling_forecasts,
        **plot_figures  # Unpack the nested plot dictionary into the main dictionary
    }
    
    return final_output

if __name__ == "__main__":
    # --- Step 1: Run Simulation or Load from Cache ---
    # Set force_rerun=True to ignore the cache and run a new simulation
    cleaned_df, all_rolling_forecasts, production_pipeline, sku_allocation_pipeline, capacity_summary = run_or_load_simulation(force_rerun=False)

    # --- Step 2: Final Outputs (using the results) ---
    print("\n--- Full Log of Category Orders Placed During Simulation ---")
    if production_pipeline:
        print(pd.DataFrame(production_pipeline).to_string())
    else:
        print("No production orders were needed during the simulation.")

    print("\n--- Full Log of SKU Allocations from Orders ---")
    sku_allocation_plan_df = pd.DataFrame(sku_allocation_pipeline)
    if not sku_allocation_plan_df.empty:
        print(sku_allocation_plan_df.to_string())
    else:
        print("No SKU allocations were made.")

    print("\n--- Daily Capacity Usage Summary ---")
    if not capacity_summary.empty:
        print(capacity_summary.to_string())
    else:
        print("No production to summarize.")

    # --- Step 3: Visualization (using the results) ---
    if all_rolling_forecasts:
        print("\n--- Generating Visualizations ---")
        unique_categories = sorted(list(set(SKU_TO_CATEGORY_MAP.values())))
        
        # SKU and Category summary plots
        for sku in config.SKUS.keys():
            plot_sku_summary(cleaned_df, all_rolling_forecasts, sku_to_plot=sku)
        for category in unique_categories:
            plot_category_summary(cleaned_df, all_rolling_forecasts, category_to_plot=category, sku_to_category_map=SKU_TO_CATEGORY_MAP)

        # Production schedule plots
        if not sku_allocation_plan_df.empty:
            for sku in config.SKUS.keys():
                plot_sku_production(sku_allocation_plan_df, sku_to_plot=sku)
            for category in unique_categories:
                plot_category_production(sku_allocation_plan_df, category_to_plot=category)