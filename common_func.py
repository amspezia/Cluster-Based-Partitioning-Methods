import numpy as np


def sort_by_label(X, y):
    sorted_index = y.argsort()
    return X[sorted_index], y[sorted_index]

def circular_append(input_list, output_list, k):
    i = 0
    for x in input_list:
        output_list[i].append(x)
        if i < k-1:
            i += 1
        else:
            i = 0
    return(output_list)

def data_slicing_by_label(X, y, indexes):
    X_out = np.split(X, indexes)
    X_out = X_out[:-1]
    y_out = np.split(y, indexes)
    y_out = y_out[:-1]

    return X_out, y_out

def aux_indexes(y):
    counts = np.unique(y, return_counts=True)
    slicing_indexes = np.cumsum(counts[1])
    segment_shift = np.append(0, slicing_indexes)
    start_of_segment = segment_shift[:-1]
    end_of_segment = segment_shift[1:]

    return slicing_indexes, segment_shift, start_of_segment, end_of_segment