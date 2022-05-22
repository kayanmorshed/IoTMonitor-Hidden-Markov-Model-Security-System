X = "dlcfw"
Y = ["dllcf", "flcfw", "dlcwf"]
S = []
E = {}
S = [['d', 'l', 'c', 'f'],['l', 'c', 'f', 'w'],['d','l', 'c', 'f'],['d','l', 'c', 'w']]
for i in range(len(Y)+1):
    #S.append(LCS(X, Y[i]))
    print(S[i])
    temp = ''.join(S[i])

    for j in range(len(temp) - 1):
        #print(i,j)
        tempKey = temp[j],temp[j+1]
        if tempKey in E:
            E[tempKey] = E[tempKey] + 1
        else:
            E[tempKey] = 1
    print("--")
print(E)