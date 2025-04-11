import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index):
        pass

class SecondOfMinute(TimeFeature):
    def __call__(self, index):
        return index.second / 59.0

class MinuteOfHour(TimeFeature):
    def __call__(self, index):
        return index.minute / 59.0

class HourOfDay(TimeFeature):
    def __call__(self, index):
        return index.hour / 23.0

class DayOfWeek(TimeFeature):
    def __call__(self, index):
        return index.dayofweek / 6.0

class DayOfMonth(TimeFeature):
    def __call__(self, index):
        return (index.day - 1) / 30.0

class DayOfYear(TimeFeature):
    def __call__(self, index):
        return (index.dayofyear - 1) / 365.0

class MonthOfYear(TimeFeature):
    def __call__(self, index):
        return (index.month - 1) / 11.0

# Move the dictionary outside all classes
time_features_dict = {
    'h': [HourOfDay(), DayOfWeek(), DayOfMonth(), MonthOfYear()],
    't': [MinuteOfHour(), HourOfDay(), DayOfWeek(), DayOfMonth(), MonthOfYear()],
    's': [SecondOfMinute(), MinuteOfHour(), HourOfDay(), DayOfWeek(), DayOfMonth(), MonthOfYear()],
}

def time_features(dates, timeenc=0, freq='h'):
    """
    Main API for time features
    """
    print(f"Input dates shape: {dates.shape}")  # Debugging statement
    print(f"Input dates sample: {dates.head()}")  # Debugging statement

    # Convert to DatetimeIndex so that attributes like hour work element-wise
    dates = pd.DatetimeIndex(pd.to_datetime(dates['timestamp']))

    if timeenc == 0:
        # Generate time features based on the frequency
        features = np.stack([
            dates.hour,         # Hour of the day
            dates.dayofweek,    # Day of the week
            dates.day,          # Day of the month
            dates.month         # Month of the year
        ], axis=1)
        print(f"Generated time features shape: {features.shape}")  # Debugging statement
        print(f"Sample time features: {features[:5]}")  # Debugging statement
        return features

    if timeenc == 1:
        # Use positional encoding
        dates = dates.map(lambda x: x.toordinal())
        return np.array(dates)[:, None]

    return dates.map(lambda x: x.weekday()).values