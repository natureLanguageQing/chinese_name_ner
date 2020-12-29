name = open("../entity/name.txt", "r", encoding="utf-8")
for i in name.readlines():
    print("https://www.baidu.com/s?ie=UTF-8&wd=" + i.strip("\n"))
