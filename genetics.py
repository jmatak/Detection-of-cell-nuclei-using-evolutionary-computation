from deap import base, creator, tools, algorithms
import numpy
import morphology_transformation
import cv2
import image_process

ALLELE_MUTATION_CHANCE = 0.1
TOURNAMENT_SIZE = 3
POPULATION_SIZE = 20
CROSS_PROBABILITY = 0.7
MUTATION_PROBABILITY = 0.3
NUMB_OF_GENERATIONS = 10
INITIAL_IND_LENGTH = 8

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


if __name__ == "__main__":
    results = main()
    print(results[2])