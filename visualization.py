import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_sales_per_year(df):
    sum_per_year = df.groupby('YEAR')['QUANTITY'].sum()
    plt.figure(figsize=(10,6))
    bars = plt.bar(sum_per_year.index, sum_per_year.values, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title('Total Sales Quantity per Year')
    plt.xlabel('Year')
    plt.ylabel('Quantity')
    for bar in bars:
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{int(bar.get_height())}', ha='center')
    plt.show()

def plot_price_change(df):
    mean_price = df.groupby('YEAR')['SALE_PRICE'].mean()
    plt.figure(figsize=(10,6))
    bars = plt.bar(mean_price.index, mean_price.values, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title('Mean Price per Year')
    plt.xlabel('Year')
    plt.ylabel('SALE_PRICE')
    for bar in bars:
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{int(bar.get_height())}', ha='center')
    plt.show()

def simple_heatmap(df):
    pivot = df.pivot_table(values='QUANTITY', index='YEAR', columns='MONTH', aggfunc='sum')
    plt.figure(figsize=(10,6))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='Blues')
    plt.show()

def simple_line_plot(df):
    monthly = df.groupby(['YEAR','MONTH'])['QUANTITY'].sum().reset_index()
    for y in df['YEAR'].unique():
        data = monthly[monthly['YEAR']==y]
        plt.plot(data['MONTH'], data['QUANTITY'], marker='o', label=str(y))
    plt.title('Monthly Sales')
    plt.xlabel('Month')
    plt.ylabel('Quantity')
    plt.legend()
    plt.show()

import matplotlib.pyplot as plt

def plot_trends_with_periods(df):

    # Ομαδοποίηση ανά ημερομηνία
    daily = df.groupby('DATE')['QUANTITY'].sum().reset_index()

    plt.figure(figsize=(15,6))

    # Plot συνολικών πωλήσεων
    plt.plot(daily['DATE'], daily['QUANTITY'], marker='o', label='Quantity Sold', color='skyblue')

    # Χρωματισμός περιοχών
    for period, color, label in [
        ('IS_COVID', 'red', 'Covid'),
        ('IS_WAR_PERIOD', 'orange', 'War'),
        ('IS_HIGH_SEASON_PERIOD', 'green', 'High Season')
    ]:
        mask = df[period] == 1
        plt.fill_between(df['DATE'], 0, df['QUANTITY'].max(), where=mask, color=color, alpha=0.2, label=label)

    plt.title("Sales Quantity Trends with Special Periods Highlighted")
    plt.xlabel("Date")
    plt.ylabel("Quantity Sold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
