import datetime
import pytz

good_time_intervals = [(datetime.datetime(2016, 7, 30, 0, 0, tzinfo=pytz.UTC), datetime.datetime(2016, 8, 2, 0, 0, tzinfo=pytz.UTC)),
    (datetime.datetime(2016, 8, 6, 0, 0, tzinfo=pytz.UTC), datetime.datetime(2016, 8, 8, 0, 0, tzinfo=pytz.UTC)),
    (datetime.datetime(2016, 8, 13, 0, 0, tzinfo=pytz.UTC), datetime.datetime(2016, 8, 17, 0, 0, tzinfo=pytz.UTC)),
    (datetime.datetime(2016, 8, 19, 0, 0, tzinfo=pytz.UTC), datetime.datetime(2016, 8, 22, 0, 0, tzinfo=pytz.UTC)),
    (datetime.datetime(2016, 8, 24, 0, 0, tzinfo=pytz.UTC), datetime.datetime(2016, 8, 25, 0, 0, tzinfo=pytz.UTC)),
    (datetime.datetime(2016, 8, 31, 0, 0, tzinfo=pytz.UTC), datetime.datetime(2016, 9, 3, 0, 0, tzinfo=pytz.UTC)),
    (datetime.datetime(2016, 9, 6, 0, 0, tzinfo=pytz.UTC), datetime.datetime(2016, 9, 18, 0, 0, tzinfo=pytz.UTC))]

def get_good_time_intervals():
    global good_time_intervals

    for (begin, end) in good_time_intervals:
        yield (begin, end)

def get_good_days(offset=0):
    offset = datetime.timedelta(days=offset)
    for (begin, end) in get_good_time_intervals():
        current_begin = begin + offset
        delta = datetime.timedelta(days=1)

        while current_begin + delta <= end - offset:
            yield current_begin, current_begin + delta
            current_begin += delta

# These bees were marked with smaller tags, leading to a higher miss rate in their detections.
bad_bee_ids = {1922, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2059, 2060, 2061, 2062, 2063, 2064,
    2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079}

def get_bad_bee_ids():
    return bad_bee_ids