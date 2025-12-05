"""
FastAPI, POST/HTTP, logging, csv, exception handling

to run in cmd prompt:
fastapi run <this .py>

then, likely need to go to (see output after running cmd above):
http://localhost:8000/docs
"""

import csv
import logging
import math
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from io import StringIO
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S',
)


app = FastAPI()

lat_km = 92
lng_km = 111


def distance(lat1, lng1, lat2, lng2):
    """Return euclidean distance (in kilometers) between two coordinates.

    >>> distance(0, 0, 1, 1)
    144.1700384962146
    """
    delta_lat = (lat1 - lat2) * lat_km
    delta_lng = (lng1 - lng2) * lng_km
    return math.hypot(delta_lat, delta_lng)


def parse_csv(fp):
    """Parse CSV, returns tuple of:
    - number of samples
    - total distance
    - average speed in km/h

    the input CSV expected to be in this format:
    time,lat,lng,height
    2024-03-19 03:48:07.235,32.519585,35.015021,136.1999969482422
    2024-03-19 03:48:24.734,32.519606,35.014954,126.5999984741211
    2024-03-19 03:48:25.660,32.519612,35.014871,123.0
    2024-03-19 03:48:26.819,32.519654,35.014824,120.5
    """
    reader = csv.DictReader(fp)
    start_time = end_time = None
    prev_lat = prev_lng = None
    count = total_distance = 0

    for row in reader:
        count += 1
        time = datetime.fromisoformat(row['time'])
        # Rows are in chronological order
        if start_time is None:
            start_time = time
        else:
            end_time = time

        lat, lng = float(row['lat']), float(row['lng'])
        if prev_lat:
            total_distance += distance(lat, lng, prev_lat, prev_lng)
        prev_lat, prev_lng = lat, lng

    duration_hours = (end_time - start_time).total_seconds() / (60 * 60)
    speed = total_distance / duration_hours

    return count, total_distance, speed


MAX_CSV_SIZE = 5 * (1 << 20)  # 5 MB


def timed(fn):
    """A decorator that logs function run time."""
    fn_name = fn.__name__

    @wraps(fn)
    async def wrapper(*args, **kw):
        start = perf_counter()
        try:
            return await fn(*args, **kw)
        finally:
            duration = perf_counter() - start
            logging.info(f'[metric:{fn_name}.time] %.3f', duration)

    return wrapper


@app.post('/run')
@timed
async def run_stats(request: Request):
    if (mime_type := request.headers['content-type']) != 'text/csv':
        logging.error(f'bad format: {mime_type}')
        raise HTTPException(
            HTTPStatus.NOT_ACCEPTABLE,
            detail='not a CSV',
        )

    if (size := int(request.headers['Content-Length'])) > MAX_CSV_SIZE:
        logging.error(f'[run_stats] file too large: {size}')
        raise HTTPException(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            detail='file too large',
        )

    data = await request.body()
    fp = StringIO(data.decode())
    count, distance, speed = parse_csv(fp)
    out = {
        'count': count,
        'distance': distance,
        'speed': speed,
    }
    logging.info('run_stats - %s', out)
    return out
