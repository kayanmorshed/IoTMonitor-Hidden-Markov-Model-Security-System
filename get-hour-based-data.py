import pandas as pd
import os


def get_until(date, sep, hour, min_sec):
    until = date + sep + str(hour) + min_sec
    return until

def get_hourly_data(csv_directory, filename):
    # read the dataset with the passed filename
    read_dataset = pd.read_csv(os.path.join(csv_directory, filename) + '.csv')

    # get the columns in separate list
    timestamp = read_dataset['timestamp'].values
    event_value = read_dataset['event_value'].values
    sensor_value = read_dataset['sensor_value'].values

    # set the threshold upto which the rows are extracted
    date = '2019-04-06'
    date_hour_separator_u10 = ' 0'
    date_hour_separator_a10 = ' '
    hour = 1
    minute_sec = ':00:00'

    # set the starting point
    start_from = 0

    until = []

    j = 0

    while hour < 25:
        # construct the date
        if hour < 10:
            # print(get_until(date, date_hour_separator_u10, hour, minute_sec))
            until.append(get_until(date, date_hour_separator_u10, hour, minute_sec))
            j += 1
        else:
            until.append(get_until(date, date_hour_separator_a10, hour, minute_sec))
            j += 1

        dframe = pd.DataFrame(data={'timestamp': [], 'event_value': [], 'sensor_value': []})
        index_dframe = 0

        for i in range(start_from, read_dataset.shape[0]):
            if timestamp[i].find(until[0]) < 0:
                dframe.loc[index_dframe] = [timestamp[i], event_value[i], sensor_value[i]]
                index_dframe += 1
            else:
                start_from = i
                break

        # convert dataframe to csv
        # dframe.to_csv(os.path.join(csv_directory, 'hour_based', filename) + '_hour_' + str(hour) + '.csv', index=False)
        dframe.to_csv('sample/hour_' + str(hour) + '.csv', index=False)

        until.pop()

        # set for the next hour
        hour += 1
    return 0


if __name__ == '__main__':
    # get the directory "dataset"
    dataset_directory = os.path.join(os.getcwd(), 'dataset_workspace', '2019-04-04', 'categorical')

    # get all the file names with the extension .csv and put into a list
    file_name_list = []

    for file_name in os.listdir(dataset_directory):
        if '.csv' in file_name:
            file_name_list.append(file_name[:-4])

    # iterate the method "conversion_into_numerical()" for all the csvs containing name in "file_name_list"
    for i in range(0, len(file_name_list)):
        get_hourly_data(dataset_directory, file_name_list[i])


