
#%%


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Convert the RGBA values from 0-255 range to 0-1 range
rgba_color = (252/255, 230/255, 14/255, 255/255)


# Set the time period
start_year = 2000
end_year = 2020
dates = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31', freq='D')

# Yearly Trend Component
yearly_trend = np.linspace(0, 1, len(dates)) * 5

# Seasonal Component
seasonal_component = np.sin(np.linspace(0, 2 * np.pi, len(dates)))

# Random Component
random_component = np.random.normal(0, 0.5, len(dates))

# Introduce specific events
# Dot-com Bubble Burst in 2001
dot_com_bubble = ((dates.year == 2001) & (dates.month >= 3)) * -2

# Financial Crisis in 2008
financial_crisis = ((dates.year == 2008) & ((dates.month >= 9) & (dates.month <= 12))) * -3

# COVID-19 Pandemic Early 2020
covid_pandemic = ((dates.year == 2020) & (dates.month <= 6)) * -4

# Combine all components with the events
time_series = yearly_trend + seasonal_component + random_component + dot_com_bubble + financial_crisis + covid_pandemic

# Convert to a pandas DataFrame
time_series_df = pd.DataFrame(time_series, index=dates, columns=['Value'])


# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(time_series_df.index, time_series_df['Value'], color=rgba_color)
plt.title('Time Series with Yearly, Seasonal, Random Components, and Real-Life Events')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

#%%
