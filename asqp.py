"""
CAPP 30121: Airline Service Quality Performance (ASQP) data

Judith Nguli

Answer questions about airline performance data.
"""
import numpy as np

DATA_LEN = 36

############## TASK1 1 ##############

def average_delay(arrival_time):
    """
    Given ASQP data, determine the average flight delay 

    Args:
        arrival_time (NumPy array): flight arrival times (negative
            if the flight is early, positive if the flight is late)

    Returns (float): the average delay of non-early flights
    """

    delayed_frame = arrival_time[arrival_time >= 15]  # that is atleast 15 minutes
    return delayed_frame.mean()


def delay_and_cancel_fractions(arrival_time, cancellation_code):
    """
    Given ASQP data, compute the fraction of delayed and cancelled flights

    Args:
        arrivals_time (NumPy array): flight arrival times
        cancellation_code (NumPy array): flight cancellation codes

    Returns (tuple of floats): fraction of flights that were delayed, 
        fraction of flights that were cancelled
    """

    delayed_flights = arrival_time[arrival_time >= 15]
    cancelled_flights = cancellation_code[~(cancellation_code == 'nan')]

    delay_ratio = delayed_flights.size / arrival_time.size
    cancelled_ratio = cancelled_flights.size / cancellation_code.size
    return delay_ratio, cancelled_ratio


def per_carrier_cancels(cancellation_code_by_carriers):
    """
    Given ASQP data, determine how many of each carrier's flights were
        cancelled, and which carrier cancelled the most flights

    Input:
        cancellation_code_by_carriers (dictionary): a dictionary mapping a 
            carrier to a NumPy array of that carrier's flight cancellation codes

    Returns (dictionary, string): a dictionary mapping a carrier to the number of
        cancelled flights, and the carrier with the most cancellations
    """

    most_cancelled, poor_carrier = -1, ""
    count_cancelled = {}
    for carrier in cancellation_code_by_carriers:
        carrier_cancellations = cancellation_code_by_carriers[carrier]
        number_cancelled = carrier_cancellations[~(carrier_cancellations == 'nan')].size

        count_cancelled[carrier] = number_cancelled
        if number_cancelled > most_cancelled:
            most_cancelled, poor_carrier = number_cancelled, carrier

    return count_cancelled, poor_carrier


def underperforming_carriers(arrivals_by_carrier):
    """
    Given ASQP data, determine which carriers have a worse than average delay 

    Input:
        arrivals_by_carrier (dictionary): a dictionary mapping a carrier to a 
            NumPy array of that carrier's arrival times

    Returns (list): carriers whose average delay is worse than the overall
        average delay
    """
    aggregate_data = None
    for k, v in arrivals_by_carrier.items():
        if aggregate_data is None:
            aggregate_data = v
        else:
            aggregate_data = np.concatenate((aggregate_data, v))

    delay_filter = aggregate_data >= 15
    valid_delays = aggregate_data[delay_filter]  # delays at least 15 minutes inclusive
    if valid_delays.size == 0:
        return []  # no carriers with any delays greater that 15 minutes
    
    av_delay = valid_delays.mean()

    poor_carriers = []
    for carrier, carrier_data in arrivals_by_carrier.items():
        carrier_valid_delays = carrier_data[carrier_data >= 15]
        carrier_mean_delay = carrier_valid_delays.mean()
        if carrier_mean_delay > av_delay:
            poor_carriers.append(carrier)

    return poor_carriers


############## TASK 2 ##############

def read_and_process_npy(filename):
    """
    Read in and process time series ASQP data

    Input:
        filename (string): name of the NPY file

    Returns (NumPy array): a time series NumPy array
    """
    ts_data = np.load(filename)
    return np.count_nonzero(ts_data >= 60, axis=1)


############## PART 2 HELPER FUNCTION ##############

def perform_least_squares(y):
    """
    Given a data set, finds the line best fit perform least squares

    Input:
        y (array): data

    Returns (tuple of floats): the slope and y-intercept of the
        line fitted to y
    """

    x = np.arange(len(y))
    A = np.vstack([x, np.ones(len(y))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]

    return m, b


############## TASK 3 ##############

def remove_irregularities(ts, width):
    """ 
    Apply a smoothing technique to remove irregularities from the 
        times series ASQP data

    Input:
        ts (NumPy array): the time series
        width (int): the width over which to smoothe

    Returns (NumPy array): smoothed time series data
    """

    smoothed_ts = np.array(ts, copy=True).astype(float)  # create a new array
    num_rows = smoothed_ts.size

    for i in range(num_rows):
        start = i - width if i - width > 0 else 0  
        end = i + width + 1 if i + width + 1 < num_rows else num_rows  
        smoothed_ts[i] = ts[start:end].mean()

    return smoothed_ts


def remove_trend(ts, width):
    """ 
    Remove overall trend from time series ASQP data

    Input:
        ts (NumPy array): the time series
        width (int): the width over which to smoothe

    Returns (NumPy array): detrended time series data
    """
    smoothed_data = remove_irregularities(ts, width)
    m, b = perform_least_squares(smoothed_data)  # the first column.
    cleaned_ts = np.array(ts, copy=True).astype(float)
    for i in range(ts.size):
        off_set = m*i
        cleaned_ts[i] = cleaned_ts[i] - off_set

    return cleaned_ts


def is_seasonal(ts, width):
    """ 
    Bucket late flights, determine the bucket with the most 
        late flights

    Input:
        ts (NumPy array): the time series
        width (int): the width over which to smoothe
        list_of_indices (lit of list of ints): something here

    Returns (NumPy array, int): number of delays in each month,
        the index of the month with the most delays (January = 0, etc.)
    """
    trendless_data = remove_trend(ts, width)  
    
    months = np.ones(12, dtype=float)
    for i in range(12):
        data = trendless_data[i:trendless_data.size:12]
        months[i] = data.sum()

    index_of = np.argmax(months)
    return months, index_of