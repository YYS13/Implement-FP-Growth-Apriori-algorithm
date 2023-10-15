from csv import reader
def DataLoader(path):
    if path[-4:] == '.txt':
        with open(path, "r") as file:
            transcation = {}
            totalItemSet = {}
            for line in file:
                d = line.split()
                if d[2] not in totalItemSet.keys():
                    totalItemSet[d[2]] = 1
                else:
                    totalItemSet[d[2]]+=1
                if d[1] not in transcation.keys():
                    transcation[d[1]] = [d[2]]
                else:
                    transcation[d[1]].append(d[2])
            return list(transcation.values()), totalItemSet
    else:
        with open(path, "r", encoding="utf-8") as file:
            transcation = {}
            totalItemSet = {}
            csv_reader = reader(file)
            for line in csv_reader:
                if line[2] not in totalItemSet.keys():
                    totalItemSet[line[2]] = 1
                else:
                    totalItemSet[line[2]]+=1
                if line[1] not in transcation.keys():
                    transcation[line[1]] = [line[2]]
                else:
                    transcation[line[1]].append(line[2])
            return list(transcation.values()), totalItemSet