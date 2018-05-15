# coding: utf8
from deap import base, creator, tools, algorithms
import image_process
# Paralelizam
from scoop import futures
import numpy
import morphology_transformation
import morpho_util
from parameters import *
import sys
import joblib
import matplotlib.pyplot as plt

# Kreiranje funkcije fitnesa i opis jedinke
# Jedinka je lista Morfoloških transformacija
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Inicijalizacija programa
toolbox.register("individual", morphology_transformation.create, creator.Individual, length=INITIAL_IND_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", morphology_transformation.evaluate)
toolbox.register("mate", morphology_transformation.cross)
toolbox.register("mutate", morphology_transformation.mutate, mutation_chance=ALLELE_MUTATION_CHANCE)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)


# Otkomentirati liniju za paralelizam (parametri: -m scoop)
# toolbox.register("map", futures.map)

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
    plt.plot(numpy.array(maxValues))
    plt.savefig('plots/kretanje_max')
    plt.close()
    plt.boxplot(numpy.array(populationMovement))
    plt.savefig('plots/kretanje_jedinke')
    plt.close()


def genetics():
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: (ind, ind.fitness.values))
    stats.register("INDIVIDUAL", printBest)
    stats.register("MED", calculate_median)
    stats.register("AVG", calculate_mean)
    stats.register("STD", calculate_std)
    stats.register("MIN", calculate_min)
    stats.register("MAX", calculate_max)
    stats.register("GRAPH", graphInfo)

    print('Program bez transformacije slike : {}'.format(morphology_transformation.evaluate([])))
    pop, logbook = algorithms.eaSimple(pop, toolbox,
                                       cxpb=CROSS_PROBABILITY,
                                       mutpb=MUTATION_PROBABILITY,
                                       ngen=NUMB_OF_GENERATIONS,
                                       stats=stats, halloffame=hof, verbose=True)

    makePlots()
    return pop, logbook, hof


def evolution():
    results = genetics()
    joblib.dump(results[2][0], "transformations/best.ind")
    view(results[2][0])


def view(ind):
    import image_viewer as iw
    iw.viewer(ind)


def main():
    if len(sys.argv) < 2:
        print("Nedovoljno ulaznih argumenata, molim odaberite -g ili -c")
        return

    if len(sys.argv) >= 3:
        if sys.argv[2] == '-s':
            morphology_transformation.compareFunction = morphology_transformation.simple_compare
        elif sys.argv[2] == '-stn':
            morphology_transformation.compareFunction = morphology_transformation.simple_compare_without_tn
        elif sys.argv[2] == '-w':
            morphology_transformation.compareFunction = morphology_transformation.compare_weighted
        else:
            print("Nevaljan ulazni parametar, molim odaberite -reg ili irr")
            return

    if len(sys.argv) >= 4:
        if sys.argv[3] == '-irr':
            morphology_transformation.kernelMutationFunction = morphology_transformation._mutate_kernel_by_value
            morphology_transformation.randomKernel = morpho_util.get_random_irregural_kernel
        elif sys.argv[3] == '-reg':
            morphology_transformation.kernelMutationFunction = morphology_transformation._mutate_kernel
            morphology_transformation.randomKernel = morpho_util.get_random_kernel
        else:
            print("Nevaljan ulazni parametar, molim odaberite -s, -stn, -w")

    if sys.argv[1] == '-g':
        view([])
    elif sys.argv[1] == '-c':
        image_process.load('images_serialized/images_with_info.dict')
        evolution()
    else:
        print("Nevaljan ulazni parametar, molim odaberite -g ili -c")


if __name__ == "__main__":
    main()
