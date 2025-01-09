import numpy as np
import pandas as pd
import pickle

# import student code
from asqp import average_delay, \
                 delay_and_cancel_fractions, \
                 per_carrier_cancels, \
                 underperforming_carriers, \
                 read_and_process_npy, \
                 remove_irregularities, \
                 remove_trend, \
                 is_seasonal

DATA_DIR = "full_data_sets/"
TEST_DIR = "full_data_sets/"

def test_task1a():
    file = "full_data_sets/asqp.asc"

    df = pd.read_csv(file, sep="|", low_memory=False)
    arrival_time = np.array(df["Arrival Delay"], dtype='int64')

    actual = average_delay(arrival_time)
    expected = 63.77200407619861
    
    assert np.isclose(actual, expected), "Value returned by average_delay not as expected"

def test_task1b():
    file = "full_data_sets/asqp.asc"

    df = pd.read_csv(file, sep="|", low_memory=False)
    arrival_delay = np.array(df["Arrival Delay"], dtype='int64')
    cancellation_code = np.array(df["Cancellation Code"], dtype=str)

    actual_delays, actual_cancels = delay_and_cancel_fractions(arrival_delay, cancellation_code)
    expected_delays = 0.1873460363600532
    expected_cancels = 0.012289377740552792

    assert np.isclose(actual_delays, expected_delays), "Value returned by delay_and_cancel_fractions not as expected"
    assert np.isclose(actual_cancels, expected_cancels), "Value returned by delay_and_cancel_fractions not as expected"

def test_task1c():
    file = "full_data_sets/asqp.asc"

    df = pd.read_csv(file, sep="|", low_memory=False)

    cancellation_codes = {}
    for i, carrier in enumerate(df["Marketing Carrier"]):
        if not carrier in cancellation_codes:
            cancellation_codes[carrier] = []
        cancellation_codes[carrier].append(df["Cancellation Code"][i])

    for carrier in cancellation_codes:
        cancellation_codes[carrier] = np.array(cancellation_codes[carrier], dtype=str)

    actual_cancellations, actual_worst = per_carrier_cancels(cancellation_codes)
    expected_cancellations = {'DL': 412, 'AA': 4356, 'AS': 459, 'UA': 1554, 'B6': 53, 'HA': 33, 'F9': 70, 'G4': 62, 'NK': 52, 'WN': 931}
    expected_worst = 'AA'

    assert actual_cancellations == expected_cancellations, "Value returned by per_carrier_cancels not as expected"
    assert actual_worst == expected_worst, "Value returned by per_carrier_cancels not as expected"

def test_task1d():
    file = "full_data_sets/asqp.asc"

    df = pd.read_csv(file, sep="|", low_memory=False)

    arrival_times = {}
    for i, carrier in enumerate(df["Marketing Carrier"]):
        if not carrier in arrival_times:
            arrival_times[carrier] = []
        arrival_times[carrier].append(df["Arrival Delay"][i])

    for carrier in arrival_times:
        arrival_times[carrier] = np.array(arrival_times[carrier], dtype='int64')

    actual_worst = underperforming_carriers(arrival_times)
    expected_worst = ['DL', 'UA', 'B6', 'F9', 'G4', 'NK']

    assert actual_worst == expected_worst, "Value returned by underperforming_carriers not as expected"

def task2_helper(actual_filename, expected_filename):

    actual = read_and_process_npy(actual_filename)
    expected = np.load(expected_filename)

    assert np.array_equal(actual, expected), "Array returned by read_and_process_npy not as expected"

def test_task2_full():
    task2_helper(DATA_DIR + "delays.npy", TEST_DIR + "task3_delays.npy")

def task3a_helper(actual_filename, expected_filename, w):

    ts = np.load(actual_filename)
    actual = remove_irregularities(ts, w)
    expected = np.load(expected_filename)

    assert np.isclose(actual, expected).all(), "Array returned by remove_irregularities not as expected"

def test_task3a_full_0():

    w = 1
    actual_filename = DATA_DIR + "task3_delays.npy"
    expected_filename = DATA_DIR + "task4a_delays_w" + str(w) + ".npy"
    task3a_helper(actual_filename, expected_filename, w)

def test_task3a_full_1():

    w = 2
    actual_filename = DATA_DIR + "task3_delays.npy"
    expected_filename = DATA_DIR + "task4a_delays_w" + str(w) + ".npy"
    task3a_helper(actual_filename, expected_filename, w)

def test_task3a_full_2():

    w = 3
    actual_filename = DATA_DIR + "task3_delays.npy"
    expected_filename = DATA_DIR + "task4a_delays_w" + str(w) + ".npy"
    task3a_helper(actual_filename, expected_filename, w)

def task3b_helper(actual_filename, expected_filename, w):

    ts = np.load(actual_filename)
    actual = remove_trend(ts, w)
    expected = np.load(expected_filename)

    assert np.isclose(actual, expected).all(), "Array returned by remove_irregularities not as expected"

def test_task3b_full_0():

    w = 1
    actual_filename = DATA_DIR + "task3_delays.npy"
    expected_filename = DATA_DIR + "task4b_delays_w" + str(w) + ".npy"
    task3b_helper(actual_filename, expected_filename, w)

def test_task3b_full_1():

    w = 2
    actual_filename = DATA_DIR + "task3_delays.npy"
    expected_filename = DATA_DIR + "task4b_delays_w" + str(w) + ".npy"
    task3b_helper(actual_filename, expected_filename, w)

def test_task3b_full_2():

    w = 1
    actual_filename = DATA_DIR + "task3_delays.npy"
    expected_filename = DATA_DIR + "task4b_delays_w" + str(w) + ".npy"
    task3b_helper(actual_filename, expected_filename, w)

def task3c_helper(actual_filename, expected_filename, w, expected_month):

    ts = np.load(actual_filename) 
    actual_buckets, actual_month = is_seasonal(ts, w)
    expected_buckets = np.load(expected_filename)

    assert np.isclose(actual_buckets, expected_buckets).all(), "Array returned by is_seasonal not as expected"
    assert actual_month == expected_month, "Month returned by is_seasonal not as expected"

def test_task3c_0():

    w = 1
    actual_filename = DATA_DIR + "task3_delays.npy"
    expected_filename = DATA_DIR + "task4c_delays_w" + str(w) + ".npy"
    expected_month = 6

    task3c_helper(actual_filename, expected_filename, w, expected_month)

def test_task3c_1():

    w = 2
    actual_filename = DATA_DIR + "task3_delays.npy"
    expected_filename = DATA_DIR + "task4c_delays_w" + str(w) + ".npy"
    expected_month = 6

    task3c_helper(actual_filename, expected_filename, w, expected_month)