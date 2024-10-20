# DEVELOPED BY : THANJIYAPPAN K
# REGISTER NUMBER : 212222240108
# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```PY
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset (replace 'file_path' with the actual path to your CSV)
file_path = 'ev.csv'
ev_data = pd.read_csv(file_path)

# Filter the dataset to focus on 'year' and 'value' columns
ev_data_filtered = ev_data[['year', 'value']].dropna()

# Aggregate the data by averaging 'value' for each year
ev_data_aggregated = ev_data_filtered.groupby('year').mean().reset_index()

# Extract the 'value' column as the time series data
time_series_values = ev_data_aggregated['value'].values

# 1. ARMA(1,1) Model for the aggregated data

# Fit the ARMA(1,1) model
arma11_model = ARIMA(time_series_values, order=(1, 0, 1))
arma11_fit = arma11_model.fit()

# Plot the fitted ARMA(1,1) time series
plt.figure(figsize=(10, 6))
plt.plot(time_series_values, label='Original Data')
plt.plot(arma11_fit.fittedvalues, label='Fitted ARMA(1,1)', color='red')
plt.title('ARMA(1,1) Fitted Process - Aggregated EV Data')
plt.xlabel('Time (Years)')
plt.ylabel('Aggregated Value')
plt.legend()
plt.grid(True)
plt.show()

# Determine the maximum number of lags based on the length of the time series
max_lags = min(len(time_series_values) // 2, 7)  # Set lags to be <= 50% of sample size

# Display ACF and PACF plots for the actual data with adjusted lags
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(time_series_values, lags=max_lags, ax=plt.gca())
plt.subplot(122)
plot_pacf(time_series_values, lags=max_lags, ax=plt.gca())
plt.suptitle('ACF and PACF for Aggregated EV Data')
plt.tight_layout()
plt.show()

# 2. ARMA(2,2) Model for the aggregated data

# Fit the ARMA(2,2) model
arma22_model = ARIMA(time_series_values, order=(2, 0, 2))
arma22_fit = arma22_model.fit()

# Plot the fitted ARMA(2,2) time series
plt.figure(figsize=(10, 6))
plt.plot(time_series_values, label='Original Data')
plt.plot(arma22_fit.fittedvalues, label='Fitted ARMA(2,2)', color='red')
plt.title('ARMA(2,2) Fitted Process - Aggregated EV Data')
plt.xlabel('Time (Years)')
plt.ylabel('Aggregated Value')
plt.legend()
plt.grid(True)
plt.show()

# Display ACF and PACF plots for the actual data with adjusted lags
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(time_series_values, lags=max_lags, ax=plt.gca())
plt.subplot(122)
plot_pacf(time_series_values, lags=max_lags, ax=plt.gca())
plt.suptitle('ACF and PACF for Aggregated EV Data')
plt.tight_layout()
plt.show()

```

### OUTPUT:
# SIMULATED ARMA(1,1) PROCESS:
![image](https://github.com/user-attachments/assets/861c284e-dd3f-4253-b11c-7e7bb502857f)


# Partial Autocorrelation:
![image](https://github.com/user-attachments/assets/a4978642-9bc7-49d5-8e8f-ad9a167e19b2)


# SIMULATED ARMA(2,2) PROCESS:
![image](https://github.com/user-attachments/assets/b8edb5e1-8da1-4a68-8d15-13230814c34b)


# Partial Autocorrelation:
![image](https://github.com/user-attachments/assets/95fef3e0-73b3-430a-a30f-4f46863d9cd1)


# RESULT:
Thus, a python program is created to fir ARMA Model successfully.
