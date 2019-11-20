fileName = ["nohup_2", "nohup_1", "nohup_4", "nohup"]
Fo = open("new nohup", "w")
for fil in fileName:
    lineNum = 0
    with open(fil) as F:
        for line in F:
            if lineNum % 10 == 0:
                Fo.write(",\t".join(line.split()))
                Fo.write("\n")
            lineNum += 1
        Fo.write("e\n")