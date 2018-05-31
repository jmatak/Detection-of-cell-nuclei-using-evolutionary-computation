import numpy
import matplotlib.pyplot as plt


def printBest(pop):
    best = max(pop, key=lambda x: x[1][0])
    return str(best[0])


maxValues = []
populationMovement = []


def graphInfo(pop):
    pop = [p[1][0] for p in pop]
    for p in pop: populationMovement.append(p)


def calculate_max(pop):
    pop = [p[1][0] for p in pop]
    m = numpy.max(pop)
    maxValues.append(m)
    return m


def calculate_min(pop):
    pop = [p[1][0] for p in pop]
    return numpy.min(pop)


def calculate_mean(pop):
    pop = [p[1][0] for p in pop]
    return numpy.mean(pop)


def calculate_std(pop):
    pop = [p[1][0] for p in pop]
    return numpy.std(pop)


def calculate_median(pop):
    pop = [p[1][0] for p in pop]
    return numpy.median(pop)


def makePlots():
    with open('data.csv', 'a') as file:
        for p in populationMovement:
            file.write('{}\n'.format(str(p)))
    plt.plot(numpy.array(maxValues))
    plt.savefig('plots/kretanje_max')
    plt.close()
    plt.boxplot(numpy.array(populationMovement))
    plt.savefig('plots/kretanje_jedinke')
    plt.close()
