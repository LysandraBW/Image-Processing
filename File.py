import csv
import numpy as np


def load(file_path, max_rows=None, float_conversion=False):
    data = []
    with open(file_path, mode='r') as file:
        lines = csv.reader(file)

        # The first line is skipped because it's the label
        skip_line = False

        for line in lines:
            # Skip 1st Line
            if not skip_line:
                skip_line = True
                continue

            data.append([])
            for l in line:
                data[-1].append(float(l) if float_conversion else l)

            # This is just for the case where I don't want to read in all the rows
            # because of the time it can take (depending on the file).
            if max_rows is not None and len(data) == max_rows:
                return data

    return data


# I don't like string literals much
LH_BRACKET = "["
RH_BRACKET = "]"
COMMA = ","
APOSTROPHE = "'"


def arr_to_string(arr):
    # This will contain the string representation of the array.
    # We begin with a simple "["
    string_arr = LH_BRACKET

    # The shape of the array.
    # I could work something out for an array that didn't have the same
    # amount of elements in each array of the array, but time is money.
    arr_shape = np.array(arr).shape

    # One-Dimensional Array
    if len(arr_shape) == 1:
        number_elements = arr_shape[0]
        for index in range(number_elements):
            # Add Element
            string_arr += APOSTROPHE + str(arr[index]) + APOSTROPHE

            # Add Comma (except for the last element)
            if index != number_elements - 1:
                string_arr += COMMA

    # Two-Dimensional Array
    else:
        rows = arr_shape[0]
        cols = arr_shape[1]
        for sub_index in range(rows):
            # Add Another "["
            string_arr += LH_BRACKET

            # Adding Elements of Sub-Array
            for index in range(cols):
                string_arr += APOSTROPHE + str(arr[sub_index][index]) + APOSTROPHE
                if index != cols - 1:
                    string_arr += COMMA

            # End With a "]"
            string_arr += RH_BRACKET
            if sub_index != rows - 1:
                string_arr += COMMA

    # End Array with a "]"
    string_arr += RH_BRACKET

    return string_arr


def string_to_arr(string, float_conversion=False, int_conversion=False):
    # This will contain the array we extract from the string.
    arr = []

    # This is a stack that stores LH brackets and the index of said bracket.
    # It's used for determining where the pairs of brackets are.
    brackets = []

    # The pairs of brackets are stored here.
    pairs = []

    # Storing the Pairs of Brackets
    for index in range(len(string)):
        char = string[index]

        # The left-hand brackets are stored
        if char == LH_BRACKET:
            brackets.append((LH_BRACKET, index))

        # When we find a right-hand bracket, we have found the compliment to the closest left-hand bracket.
        # The closest left-hand bracket will be the top of the bracket (stack). So we store the pair, and pop the left-hand bracket
        # from the stack.
        elif char == RH_BRACKET:
            pairs.append((brackets[-1][1], index))
            brackets.pop()

    # There's only one pair therefore this is a one-dimensional array. The starting and ending character of the string
    # should be the LH and RH bracket, so I splice the string accordingly.
    if len(pairs) == 1:
        arr = string[1:-1].split(COMMA)
        for index in range(len(arr)):
            # Convert values if needed
            arr[index] = float(arr[index][1:-1]) if float_conversion else int(arr[index][1:-1]) if int_conversion else arr[index][1:-1]
    else:
        # The last pair are the brackets of the array that contains all the sub-arrays, and that's not needed.
        pairs.pop()

        for p in pairs:
            start, end = p[0] + 1, p[1]
            arr.append(string[start:end].split(COMMA))

            # For the array we just appended
            for index in range(len(arr[-1])):
                # Convert values (if needed)
                arr[-1][index] = float(arr[-1][index][1:-1]) if float_conversion else int(arr[-1][index][1:-1]) if int_conversion else arr[-1][index][1:-1]

    return arr


