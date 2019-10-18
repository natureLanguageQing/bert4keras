import pandas as pd

train_data = pd.read_csv("Train_DataSet.csv")
train_labels = pd.read_csv("Train_DataSet_Label.csv")
train_data_label = []
for data in train_data.values.tolist():
    for label_one in train_labels.values.tolist():
        if data[0] == label_one[0]:
            train_data_label += [[data[1], data[2], label_one[1]]]
test = pd.DataFrame(data=train_data_label)  # 数据有三列，列名分别为one,two,three
test.to_csv('test_csv_message.csv', encoding='utf-8')
print("转换成功")
