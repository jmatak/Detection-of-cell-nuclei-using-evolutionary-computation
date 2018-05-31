# coding: utf8
from deap import base, creator, tools, algorithms
import image_process
# Paralelizam
from scoop import futures
import morphology_transformation
import morpho_util
from parameters import *
import sys
import joblib
import util


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


def genetics():
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: (ind, ind.fitness.values))
    stats.register("INDIVIDUAL", util.printBest)
    stats.register("MED", util.calculate_median)
    stats.register("AVG", util.calculate_mean)
    stats.register("STD", util.calculate_std)
    stats.register("MIN", util.calculate_min)
    stats.register("MAX", util.calculate_max)
    stats.register("GRAPH", util.graphInfo)

    print('Program bez transformacije slike : {}'.format(morphology_transformation.evaluate([])))
    pop, logbook = algorithms.eaSimple(pop, toolbox,
                                       cxpb=CROSS_PROBABILITY,
                                       mutpb=MUTATION_PROBABILITY,
                                       ngen=NUMB_OF_GENERATIONS,
                                       stats=stats, halloffame=hof, verbose=True)

    util.makePlots()
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

    if '-s' in sys.argv:
        morphology_transformation.compareFunction = morphology_transformation.simple_compare
    elif '-stn' in sys.argv:
        morphology_transformation.compareFunction = morphology_transformation.simple_compare_without_tn
    elif '-w' in sys.argv:
        morphology_transformation.compareFunction = morphology_transformation.compare_weighted

    if '-irr' in sys.argv:
        morphology_transformation.kernelMutationFunction = morphology_transformation._mutate_kernel_by_value
        morphology_transformation.randomKernel = morpho_util.get_random_irregural_kernel
    elif '-reg' in sys.argv:
        morphology_transformation.kernelMutationFunction = morphology_transformation._mutate_kernel
        morphology_transformation.randomKernel = morpho_util.get_random_kernel

    if sys.argv[1] == '-g':
        view([])
    elif sys.argv[1] == '-c':
        image_process.load('images_serialized/images_with_info.dict')
        evolution()
    else:
        print("Nevaljan ulazni parametar, molim odaberite -g ili -c")


if __name__ == "__main__":
    main()
