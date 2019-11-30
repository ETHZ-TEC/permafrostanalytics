import requests
from datetime import datetime, timedelta
import pandas as pd

# url = 'https://api.darksky.net/forecast/385501f54c2d8b4545cf03182c7c8237/45.9765,7.6584,2017-08-31T00:00:00Z'
# response = requests.get(url, {'units': 'si'})

def url_by_date(date):
    date_str = datetime.strftime(date, "%Y-%m-%d")
    url = f'https://api.darksky.net/forecast/385501f54c2d8b4545cf03182c7c8237/45.9765,7.6584,{date_str}T00:00:00Z'
    return url

def make_request(date):
    response = requests.get(url_by_date(date), {'units': 'si'})
    return response.text

def save_request(date):
    with open('weatherdata/darksky-' + datetime.strftime(date, "%Y-%m-%d") + ".json", 'w') as f:
        data = make_request(date)
        f.write(data)

def save_all():
    date = datetime(2017, 1, 1)

    while date < datetime(2018, 1, 1):
        save_request(date)
        date = date + timedelta(days=1)
