name_entity = []

name = open("../entity/name.txt", "r", encoding="utf-8")
for i in name.readlines():
    name_entity.append(i.strip("\n"))
name_entity = list(set(name_entity))
