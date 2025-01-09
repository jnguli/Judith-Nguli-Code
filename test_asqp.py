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

# might need a helper to compare dfs
DATA_DIR = "data/"
TEST_DIR = "test-data/"

######### Testing Part 1: DELAYS AND CANCELLATIONS #########

def task1a_helper(file_name, expected):

    df = pd.read_csv(file_name, sep="|", low_memory=False)
    arrival_time = np.array(df["Arrival Delay"], dtype='int64')

    actual = average_delay(arrival_time)
    assert np.isclose(actual, expected), "Value returned by average_delay not as expected"

def test_task1a_0():
    task1a_helper("data/tiny.asc", 86.0)

def test_task1a_1():
    task1a_helper("data/tiny_all_delayed.asc", 62.5)

def task1b_helper(file_name, expected_delays, expected_cancels):
    
    df = pd.read_csv(file_name, sep="|", low_memory=False)
    arrival_delay = np.array(df["Arrival Delay"], dtype='int64')
    cancellation_code = np.array(df["Cancellation Code"], dtype=str)

    actual_delays, actual_cancels = delay_and_cancel_fractions(arrival_delay, cancellation_code)
    assert np.isclose(actual_delays, expected_delays), "Value returned by delay_and_cancel_fractions not as expected"
    assert np.isclose(actual_cancels, expected_cancels), "Value returned by delay_and_cancel_fractions not as expected"

def test_task1b_0():
    task1b_helper("data/tiny.asc", 0.2, 0.1)

def test_task1b_1():
    task1b_helper("data/tiny_all_on_time.asc", 0.0, 0.0)

def test_task1b_2():
    task1b_helper("data/tiny_all_delayed.asc", 1.0, 0.0)

def test_task1b_3():
    task1b_helper("data/tiny_all_cancelled.asc", 0.0, 1.0)

def task1c_helper(file_name, expected_cancellations, expected_worst):

    df = pd.read_csv(file_name, sep="|", low_memory=False)

    cancellation_codes = {}
    for i, carrier in enumerate(df["Marketing Carrier"]):
        if not carrier in cancellation_codes:
            cancellation_codes[carrier] = []
        cancellation_codes[carrier].append(df["Cancellation Code"][i])

    for carrier in cancellation_codes:
        cancellation_codes[carrier] = np.array(cancellation_codes[carrier], dtype=str)

    actual_cancellations, actual_worst = per_carrier_cancels(cancellation_codes)
    assert actual_cancellations == expected_cancellations, "Value returned by per_carrier_cancels not as expected"
    assert actual_worst == expected_worst, "Value returned by per_carrier_cancels not as expected"

def test_task1c_0():
    task1c_helper("data/tiny.asc", {'AA': 0, 'WN': 0, 'DL': 0, 'B6': 0, 'UA': 0, 'AS': 1}, 'AS')

def test_task1c_1():
    task1c_helper("data/tiny_all_cancelled.asc", {'WN': 3, 'AA': 3, 'UA': 2, 'DL': 1, 'AS': 1}, 'WN')

def task1d_helper(file_name, expected_worst):

    df = pd.read_csv(file_name, sep="|", low_memory=False)

    arrival_times = {}
    for i, carrier in enumerate(df["Marketing Carrier"]):
        if not carrier in arrival_times:
            arrival_times[carrier] = []
        arrival_times[carrier].append(df["Arrival Delay"][i])

    for carrier in arrival_times:
        arrival_times[carrier] = np.array(arrival_times[carrier], dtype='int64')

    actual_worst = underperforming_carriers(arrival_times)
    assert actual_worst == expected_worst, "Value returned by underperforming_carriers not as expected"

def test_task1d_0():
    task1d_helper("data/tiny.asc", ['WN'])

def test_task1d_1():
    task1d_helper("data/tiny_all_delayed.asc", ['AA', 'WN', 'DL'])


######### Testing Part 2: READING NPY FILE #########

def task2_helper(actual_filename, expected_filename):

    actual = read_and_process_npy(actual_filename)
    expected = np.load(expected_filename)

    assert np.array_equal(actual, expected), "Array returned by read_and_process_npy not as expected"

def test_task2_0():
    task2_helper(DATA_DIR + "delays_tiny1.npy", TEST_DIR + "task3_delays_tiny1.npy")

def test_task2_1():
    task2_helper(DATA_DIR + "delays_tiny2.npy", TEST_DIR + "task3_delays_tiny2.npy")


######### Testing Part 2: TIME SERIES #########

def task3a_helper(actual_filename, expected_filename, w):

    ts = np.load(actual_filename)
    actual = remove_irregularities(ts, w)
    expected = np.load(expected_filename)

    assert np.isclose(actual, expected).all(), "Array returned by remove_irregularities not as expected"

def test_task3a_0():

    w = 1
    actual_filename = "test-data/task3_delays_tiny1.npy"
    expected_filename = "test-data/task4a_delays_tiny1_w" + str(w) + ".npy"
    task3a_helper(actual_filename, expected_filename, w)

def test_task3a_1():

    w = 2
    actual_filename = "test-data/task3_delays_tiny1.npy"
    expected_filename = "test-data/task4a_delays_tiny1_w" + str(w) + ".npy"
    task3a_helper(actual_filename, expected_filename, w)

def test_task3a_2():

    w = 3
    actual_filename = "test-data/task3_delays_tiny1.npy"
    expected_filename = "test-data/task4a_delays_tiny1_w" + str(w) + ".npy"
    task3a_helper(actual_filename, expected_filename, w)

def test_task3a_3():

    w = 1
    actual_filename = "test-data/task3_delays_tiny2.npy"
    expected_filename = "test-data/task4a_delays_tiny2_w" + str(w) + ".npy"
    task3a_helper(actual_filename, expected_filename, w)

def test_task3a_4():

    w = 2
    actual_filename = "test-data/task3_delays_tiny2.npy"
    expected_filename = "test-data/task4a_delays_tiny2_w" + str(w) + ".npy"
    task3a_helper(actual_filename, expected_filename, w)

def test_task3a_5():

    w = 3
    actual_filename = "test-data/task3_delays_tiny2.npy"
    expected_filename = "test-data/task4a_delays_tiny2_w" + str(w) + ".npy"
    task3a_helper(actual_filename, expected_filename, w)

def task3b_helper(actual_filename, expected_filename, w):

    ts = np.load(actual_filename)
    actual = remove_trend(ts, w)
    expected = np.load(expected_filename)

    assert np.isclose(actual, expected).all(), "Array returned by remove_trend not as expected"


def test_task3b_0():

    w = 1
    actual_filename = "test-data/task3_delays_tiny1.npy"
    expected_filename = "test-data/task4b_delays_tiny1_w" + str(w) + ".npy"
    task3b_helper(actual_filename, expected_filename, w)

def test_task3b_1():

    w = 2
    actual_filename = "test-data/task3_delays_tiny1.npy"
    expected_filename = "test-data/task4b_delays_tiny1_w" + str(w) + ".npy"
    task3b_helper(actual_filename, expected_filename, w)

def test_task3b_2():

    w = 1
    actual_filename = "test-data/task3_delays_tiny2.npy"
    expected_filename = "test-data/task4b_delays_tiny2_w" + str(w) + ".npy"
    task3b_helper(actual_filename, expected_filename, w)

def test_task3b_3():

    w = 2
    actual_filename = "test-data/task3_delays_tiny2.npy"
    expected_filename = "test-data/task4b_delays_tiny2_w" + str(w) + ".npy"
    task3b_helper(actual_filename, expected_filename, w)

def task3c_helper(actual_filename, expected_filename, w, expected_month):

    ts = np.load(actual_filename) 
    actual_buckets, actual_month = is_seasonal(ts, w)
    expected_buckets = np.load(expected_filename)

    assert np.isclose(actual_buckets, expected_buckets).all(), "Array returned by is_seasonal not as expected"
    assert actual_month == expected_month, "Month returned by is_seasonal not as expected"

def test_task3c_0():

    w = 1
    actual_filename = "test-data/task3_delays_tiny1.npy"
    expected_filename = "test-data/task4c_delays_tiny1_w" + str(w) + ".npy"
    expected_month = 0

    task3c_helper(actual_filename, expected_filename, w, expected_month)

def test_task3c_1():

    w = 2
    actual_filename = "test-data/task3_delays_tiny1.npy"
    expected_filename = "test-data/task4c_delays_tiny1_w" + str(w) + ".npy"
    expected_month = 0

    task3c_helper(actual_filename, expected_filename, w, expected_month)

def test_task3c_2():

    w = 1
    actual_filename = "test-data/task3_delays_tiny2.npy"
    expected_filename = "test-data/task4c_delays_tiny2_w" + str(w) + ".npy"
    expected_month = 2

    task3c_helper(actual_filename, expected_filename, w, expected_month)

def test_task3c_3():

    w = 2
    actual_filename = "test-data/task3_delays_tiny2.npy"
    expected_filename = "test-data/task4c_delays_tiny2_w" + str(w) + ".npy"
    expected_month = 2

    task3c_helper(actual_filename, expected_filename, w, expected_month)