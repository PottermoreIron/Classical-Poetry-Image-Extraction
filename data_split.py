import json
from sklearn.model_selection import train_test_split

file_path = r'data/total.json'
data_file = open(file_path, 'r', encoding='utf-8')
total_data = json.load(data_file)
train_data, test_data = train_test_split(total_data, train_size=0.8, random_state=1)
train_path = r'data/train.json'
test_path = r'data/test.json'
with open(train_path, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open(test_path, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)
