import pandas as pd

# pd.read_csv("2020-12-29-13-10-50-73903813363400-任爱兰_百度搜索-采集的数据-后羿采集器.csv").drop_duplicates().to_csv(
#                                                                                                     index=False)
name_data = pd.read_csv("breath/breath.csv").values.tolist()
name_words_data = []
for name_one in name_data:
    name_words_data.append(name_one[4])
pd.DataFrame(name_words_data).drop_duplicates().to_csv("breath/breath.answer.csv", index=False,
                                                       header=["breath_answer"])
