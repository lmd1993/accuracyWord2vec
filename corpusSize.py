sumSize = 0
with open ("vocab.txt") as f:
    for line in f:
        fre = int(line.split('\t')[1])
        if fre <5 :
            print(line)
        sumSize+= fre
print(sumSize)
