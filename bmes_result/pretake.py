import json

bmes_train = json.load(open("../labels/bmes_train.rbtl.labels.json", encoding="utf-8"))
breath = json.load(open("../predict/breath.answer.json", encoding="utf-8"))
for breath_one in breath:
    export = list()
    for entity in breath_one["entities"]:
        if entity[0] + "-" + bmes_train[entity[1]] not in export:
            export.append(entity[0] + "-" + bmes_train[entity[1]])
    breath_one["entities"] = export
json.dump(breath, open("breath.result.json", "w", encoding="utf-8"), ensure_ascii=False)
