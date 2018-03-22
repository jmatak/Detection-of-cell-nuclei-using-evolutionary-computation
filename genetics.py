from deap import base, creator, tools, algorithms
import image_process
#Paralelizam
from scoop import futures
import numpy
import morphology_transformation
from parameters import *
import joblib

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

#Otkomentirati liniju za paralelizam (parametri: -m scoop)
# toolbox.register("map", futures.map)


def main():
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    pop, logbook = algorithms.eaSimple(pop, toolbox,
                                       cxpb=CROSS_PROBABILITY,
                                       mutpb=MUTATION_PROBABILITY,
                                       ngen=NUMB_OF_GENERATIONS,
                                       stats=stats, halloffame=hof, verbose=True)

    return pop, logbook, hof

def evolution():
    results = main()
    joblib.dump(results[2][0], "best.ind")
    view(results[2][0])

def view(ind):
    import image_viewer as iw
    iw.viewer(ind)

if __name__ == "__main__":
    evolution()