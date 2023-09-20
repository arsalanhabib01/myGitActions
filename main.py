# import required libraries
import glob
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# path to where ML files are stored
# where there are csv files for each day containing different intrusions.
path = 'C:/Users/habibars/Downloads/Network monitoring/intrusion_detection/archive/MachineLearningCVE'
all_files = glob.glob(f"{path}/*.csv")

# concatenate the 8 files into 1 
dataset = pd.concat([pd.read_csv(f) for f in all_files])


col_names = ["Destination_Port",
             "Flow_Duration", 
             "Total_Fwd_Packets", 
             "Total_Backward_Packets",
             "Total_Length_of_Fwd_Packets", 
             "Total_Length_of_Bwd_Packets", 
             "Fwd_Packet_Length_Max", 
             "Fwd_Packet_Length_Min", 
             "Fwd_Packet_Length_Mean", 
             "Fwd_Packet_Length_Std",
             "Bwd_Packet_Length_Max", 
             "Bwd_Packet_Length_Min", 
             "Bwd_Packet_Length_Mean", 
             "Bwd_Packet_Length_Std",
             "Flow_Bytes_s", 
             "Flow_Packets_s", 
             "Flow_IAT_Mean", 
             "Flow_IAT_Std", 
             "Flow_IAT_Max", 
             "Flow_IAT_Min",
             "Fwd_IAT_Total", 
             "Fwd_IAT_Mean", 
             "Fwd_IAT_Std", 
             "Fwd_IAT_Max", 
             "Fwd_IAT_Min", 
             "Bwd_IAT_Total", 
             "Bwd_IAT_Mean", 
             "Bwd_IAT_Std", 
             "Bwd_IAT_Max", 
             "Bwd_IAT_Min", 
             "Fwd_PSH_Flags", 
             "Bwd_PSH_Flags", 
             "Fwd_URG_Flags", 
             "Bwd_URG_Flags", 
             "Fwd_Header_Length", 
             "Bwd_Header_Length", 
             "Fwd_Packets_s", 
             "Bwd_Packets_s", 
             "Min_Packet_Length", 
             "Max_Packet_Length", 
             "Packet_Length_Mean", 
             "Packet_Length_Std", 
             "Packet_Length_Variance", 
             "FIN_Flag_Count", 
             "SYN_Flag_Count", 
             "RST_Flag_Count", 
             "PSH_Flag_Count", 
             "ACK_Flag_Count", 
             "URG_Flag_Count", 
             "CWE_Flag_Count", 
             "ECE_Flag_Count", 
             "Down_Up_Ratio", 
             "Average_Packet_Size", 
             "Avg_Fwd_Segment_Size", 
             "Avg_Bwd_Segment_Size", 
             "Fwd_Header_Length", 
             "Fwd_Avg_Bytes_Bulk", 
             "Fwd_Avg_Packets_Bulk", 
             "Fwd_Avg_Bulk_Rate", 
             "Bwd_Avg_Bytes_Bulk", 
             "Bwd_Avg_Packets_Bulk",
             "Bwd_Avg_Bulk_Rate", 
             "Subflow_Fwd_Packets", 
             "Subflow_Fwd_Bytes", 
             "Subflow_Bwd_Packets", 
             "Subflow_Bwd_Bytes", 
             "Init_Win_bytes_forward", 
             "Init_Win_bytes_backward", 
             "act_data_pkt_fwd", 
             "min_seg_size_forward", 
             "Active_Mean", 
             "Active_Std", 
             "Active_Max", 
             "Active_Min", 
             "Idle_Mean", 
             "Idle_Std", 
             "Idle_Max", 
             "Idle_Min", 
             "Label" 
            ]

#Inspect the Dataset
# Assign the column names
dataset.columns = col_names


# Flow_Bytes_s, Flow_Packets_s are of type object, the rest apart from attack are numeric. However, the data inside these are numeric 
# so will convert them. Also, they have Fwd_Header_Length twice so drop the second occurence.
dataset['Flow_Bytes_s'] = dataset['Flow_Bytes_s'].astype('float64')
dataset['Flow_Packets_s'] = dataset['Flow_Packets_s'].astype('float64')
dataset = dataset.loc[:, ~dataset.columns.duplicated()]

# Remove NaN/Null/Inf Values

# check if there are any Null values
if dataset.isnull().any().any():
    # Replace Inf values with NaN
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    # Drop all occurences of NaN
    dataset = dataset.dropna()

# Distribution of Dataset
# There are only 11, 21, and 36 instances of Heartbleed, SQL injection and infiltration respectively. So, we will drop these 
# since there will not be sufficient trianing data. In addition, rename the web attacks to remove the unicode ?
dataset = dataset.replace(['Heartbleed', 'Web Attack � Sql Injection', 'Infiltration'], np.nan)
dataset = dataset.dropna()

dataset.loc[dataset.Label == 'Web Attack � Brute Force', ['Label']] = 'Brute Force'
dataset.loc[dataset.Label == 'Web Attack � XSS', ['Label']] = 'XSS'

# Split Data
# Split data using 80:20 ratio, for training and test dataset. 
# xs=feature vectors, ys=labels
xs = dataset.drop('Label', axis=1)
ys = dataset['Label']

# split dataset - stratified
x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2, random_state=0, stratify=ys)


column_names = np.array(list(x_train))
to_drop = []
for x in column_names:
    size = x_train.groupby([x]).size()
    # check for columns that only take one value
    if (len(size.unique()) == 1):
        to_drop.append(x)
to_drop

# Drop these because they only contain one value, and so are redundant as columns
x_train = x_train.drop(to_drop, axis=1)
x_test = x_test.drop(to_drop, axis=1)
dataset_copy = dataset.drop(to_drop, axis=1)

# Apply Normalisation
# Using minmax normalisation
min_max_scaler = MinMaxScaler().fit(x_train)

# Apply normalisation to dataset
x_train = min_max_scaler.transform(x_train)
x_test = min_max_scaler.transform(x_test)
############## End of Data preprocessing

# model creation
classifier =  RandomForestClassifier(n_estimators=25, max_depth=20, 
                                     min_samples_split=5, min_samples_leaf=1)
    
rfc = classifier.fit(x_train, y_train)


# Dump the trained model into a pickle file
with open("C:/Users/habibars/Downloads/Network monitoring/intrusion_detection/Random forest model/model/random_forest_model.pkl", "wb") as f:
    pickle.dump(rfc, f)


# Note: Command to run the file "python main.py"
