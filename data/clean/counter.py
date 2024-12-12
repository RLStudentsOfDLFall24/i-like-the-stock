def countstuff(file):
    f = open(file ,'r')

    counter = {}
    counter[0] = 0
    counter[1] = 0
    counter[2] = 0

    total = 0

    for line in f:
        num = int(line)
        counter[num] += 1
        total += 1

    print(file)

    for k in counter.keys():
        print(k, counter[k]/total)
    #print(counter)


tickers = ['atnf','cycc','vtak','bivi']

for t in tickers:
    countstuff(t+'_target_basic.csv')
