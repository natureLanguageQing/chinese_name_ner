import pandas as pd

# pd.read_csv("2020-12-29-13-10-50-73903813363400-任爱兰_百度搜索-采集的数据-后羿采集器.csv").drop_duplicates().to_csv(
#                                                                                                     index=False)
name_data = pd.read_csv("name_data.csv").values.tolist()
name_words_data = []
for name_one in name_data:
    if name_one[0] not in name_words_data:
        name_words_data.append(name_one[0])

    if name_one[2] not in name_words_data:
        name_words_data.append(name_one[2])
pd.DataFrame(name_words_data).drop_duplicates().to_csv("name_data_words.csv", index=False)
