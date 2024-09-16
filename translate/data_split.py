import pandas as pd
from sklearn.model_selection import train_test_split

# 读取原始数据文件
data = pd.read_json('/Users/simo/PycharmProjects/translationLearn/translate/data/translation2019zh_valid_large.json',
                    lines=True)

SEED = 1234

# 划分数据集
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=SEED)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=SEED)

# 保存为 JSON 文件
train_data.to_json('/Users/simo/PycharmProjects/translationLearn/translate/data/translation2019zh_train.json',
                   orient='records', lines=True, force_ascii=False)
valid_data.to_json('/Users/simo/PycharmProjects/translationLearn/translate/data/translation2019zh_valid.json',
                   orient='records', lines=True, force_ascii=False)
test_data.to_json('/Users/simo/PycharmProjects/translationLearn/translate/data/translation2019zh_test.json',
                  orient='records', lines=True, force_ascii=False)