def get_day_of_year_365(date_series):
    doy = date_series.dt.dayofyear
    is_leap_year = date_series.dt.is_leap_year
    is_after_feb28 = (date_series.dt.month > 2) | ((date_series.dt.month == 2) & (date_series.dt.day > 28))
    doy = doy - (is_leap_year & is_after_feb28).astype(int) - 1
    is_feb29 = (date_series.dt.month == 2) & (date_series.dt.day == 29)
    doy = doy.where(~is_feb29, 58)
    return doy
