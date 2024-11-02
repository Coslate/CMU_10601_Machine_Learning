import numpy as np


#---------------Testing Execution---------------#
if __name__ == '__main__':
    # Define the data types for each column. The first column is a string ('U10' specifies a Unicode string of up to 10 characters)
    dtype = [('Region', 'U10'), ('FICO_Score', 'f8'), ('Savings_Rate', 'f8'), ('Credit_History', 'i4'), ('Label', 'i4')]

    # Read the CSV file using genfromtxt, skipping the first row with headers
    datas = np.genfromtxt('fairness_dataset.csv', delimiter=',', skip_header=2, dtype=dtype)

    # Processing
    result = []
    thresh = 198.09
    err = 0
    for data in datas:
        avg_val = np.mean([data['FICO_Score'], data['Savings_Rate'], data['Credit_History']])
        if avg_val > thresh:
            result.append(1)
        else:
            result.append(0)
        if result[-1] != data['Label']:
            err += 1
    
    # Write out to csv file.
    result = np.array(result).reshape(-1, 1)
    data_array = np.array([list(map(str, row)) for row in datas])
    data_with_predictions = np.hstack((data_array, result.astype(str))) 
    header = "Region,FICO Score,Savings Rate (%),Credit History (months),Label,Predicted Label"
    np.savetxt('fairness_dataset_with_predictions.csv', data_with_predictions, delimiter=',', header=header, fmt='%s', comments='')
    print(f"err = {err}, err_rate = {err/len(datas)*100}%")
