import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet1", default='/Users/nia/Desktop/exploration/ML-batories/ML-Battories/DataSet/a.xlsx', help="Sheet name or index (default: 0)")
    ap.add_argument("--sheet2", default='/Users/nia/Desktop/exploration/ML-batories/ML-Battories/DataSet/b.xlsx', help="Sheet name or index (default: 0)")
    args = ap.parse_args()

    # Read entire sheet with no header so we can interpret row 1/2 explicitly
    sheet1 = '/Users/nia/Desktop/exploration/ML-batories/ML-Battories/DataSet/a.xlsx'
    sheet2 = '/Users/nia/Desktop/exploration/ML-batories/ML-Battories/DataSet/b.xlsx'


    # list for all data for classification
    all_data_classification = [] #9986x62
    all_output_classification = [] #9986
    all_regression_a_in = [] #5843x62
    all_regression_a_out= [] #5843x7
    all_regression_b_in = [] #4143x62
    all_regression_b_out= [] #4143x4


    data_a = pd.read_excel(sheet1)
    row, col = data_a.shape
    seleted_a_in, selected_a_out = [1,3,5,6], [9,10,11,12,13,14,15]
    for i in range(0,row,30):
        single_flag = True
        for j in range(30):
            if single_flag == True:
                data_i = []
                data_i.extend([data_a.iloc[i+j,1], data_a.iloc[i+j,3]])
                single_flag = False
                all_output_classification.extend([data_a.iloc[i+j,7]])
                all_regression_a_out.append([data_a.iloc[i+j,9], data_a.iloc[i+j,10], data_a.iloc[i+j,11], 
                                             data_a.iloc[i+j,12], data_a.iloc[i+j,13], data_a.iloc[i+j,14], data_a.iloc[i+j,15]])
            data_i.extend([data_a.iloc[i+j,5], data_a.iloc[i+j,6]])
        all_data_classification.append(data_i)
        all_regression_a_in.append(data_i)

    data_b = pd.read_excel(sheet2)
    row, col = data_b.shape
    seleted_b_in, selected_b_out = [1,3,5,6], [9,10,11,12]
    for i in range(0,row,30):
        single_flag = True
        for j in range(30):
            if single_flag == True:
                data_i = []
                data_i.extend([data_b.iloc[i+j,1], data_b.iloc[i+j,3]])
                single_flag = False
                all_output_classification.extend([data_b.iloc[i+j,7]])
                all_regression_b_out.append([data_b.iloc[i+j,9], data_b.iloc[i+j,10], data_b.iloc[i+j,11], data_b.iloc[i+j,12]])
            data_i.extend([data_b.iloc[i+j,5], data_b.iloc[i+j,6]])
        all_data_classification.append(data_i)
        all_regression_b_in.append(data_i)
    
    def minmax_norm(lst,print_out=False):
        arr = np.asarray(lst, dtype=float)
        lo, hi = np.min(arr), np.max(arr)
        if np.isclose(hi, lo):         # all values equal → return zeros
            return np.zeros_like(arr)
        if print_out:
            print("Low, high norm: {}, {} ".format(lo, hi))
        return (arr - lo) / (hi - lo)
    # list to np array
    all_data_classification = minmax_norm(all_data_classification,1) #9986x62
    all_output_classification = np.array(all_output_classification) #9986
    all_output_classification[all_output_classification==1] = 0
    all_output_classification[all_output_classification==3] = 1

    all_regression_a_in = minmax_norm(all_regression_a_in,1)  #5843x62
    all_regression_a_out = minmax_norm(all_regression_a_out,1) #5843x7
    all_regression_b_in = minmax_norm(all_regression_b_in,1)  #4143x62
    all_regression_b_out = minmax_norm(all_regression_b_out,1) #4143x4

    np.savez('DataSet/classification-data/data.npz',x=all_data_classification,y=all_output_classification)
    np.savez('DataSet/regression-data-a/data.npz',x=all_regression_a_in,y=all_regression_a_out)
    np.savez('DataSet/regression-data-b/data.npz',x=all_regression_b_in, y=all_regression_b_out)





if __name__ == "__main__":
    main()

