import pandas as pd
import os


if __name__ == '__main__':

    # file_directory = os.path.join(os.getcwd(), 'length-based', 'v1')
    file_directory = os.path.join(os.getcwd(), 'c4')

    for filename in [filename for filename in os.listdir(file_directory) if '.csv' in filename]:
        print(filename)

        # read tge performance_eval dataset
        input_eval_dst = pd.read_csv(os.path.join(file_directory, filename))

        # read the dataset names existed in a window
        entry_dst_name= input_eval_dst['dataset_name']

        # print(entry_dst_name[0])
        # print(entry_dst_name[0][13:-14])
        # print(len(entry_dst_name[0]))

        # empty list
        length_ = []

        # populate the length list with zero
        for i in range(len(entry_dst_name)):
            length_.append(0)

        idx = 0
        for name in entry_dst_name:

            # ******************** for window sizes in 2 digits *********************
            # if name[13:14] == '_':
            #     length_[idx] = int(name[15:-14])  # 1 digit length
            # elif name[13:14] == 'l':
            #     length_[idx] = int(name[14:-14]) # 2 digit length
            # else:
            #     length_[idx] = int(name[13:-14]) # 3 digit length

            # ********************* for window sizes in 3 digits *********************
            if name[13:14] == '_':
                length_[idx] = int(name[15:-14])  # 1 digit length
            elif name[13:14] == 'l':
                length_[idx] = int(name[14:-14]) # 2 digit length
            else:
                length_[idx] = int(name[13:-14]) # 3 digit length

            idx += 1

        # create a dataframe
        dframe = pd.DataFrame(data={'length': length_})

        # concatenate dframe with input_eval_dst
        df = pd.concat([input_eval_dst, dframe], axis=1)

        # sort the newly create dataframe
        df_sorted = df.sort_values('length')

        print(df_sorted.head())

        df_sorted.to_csv(os.path.join(file_directory, filename[:-4] + '_sorted' + '.csv'), index=False)



