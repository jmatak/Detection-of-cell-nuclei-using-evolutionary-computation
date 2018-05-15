import copy
import image_process
from parameters import *
import morphology_transformation
import structuring_elements
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import numpy as np
import cv2


class ImageProcessor(object):
    """
    Klasa koja služi za procesuiranje slike transformacijama nastalim iz stablaste strukture.

    """

    def _reset(self, original, mask):
        self.image = copy.copy(original)
        self.dominantColor1 = image_process.kmeans(original, 1)
        self.dominantColor2 = image_process.kmeans(original, 2)
        self.mask = mask

    def dominantColor2Quadraple(self, out1, out2, out3, out4, image=None):
        """
        Odluka izlaza na temelju dvije dominantne boje unutar slike
        """
        if out1:
            if np.allclose(self.dominantColor2, np.array([[64], [64]]), atol=64):
                self.image = out1(image=self.image)
            elif np.allclose(self.dominantColor2, np.array([[64], [192]]), atol=64):
                self.image = out2(image=self.image)
            elif np.allclose(self.dominantColor2, np.array([[192], [64]]), atol=64):
                self.image = out3(image=self.image)
            elif np.allclose(self.dominantColor2, np.array([[192], [192]]), atol=64):
                self.image = out4(image=self.image)


    def dominantColor1Bipolar(self, out1, out2, image=None):
        """
        Odluka izlaza na temelju dominantne boje unutar slike, odluka je tamna ili svijetla pozadina
        """
        if out1:
            if np.allclose(self.dominantColor1, np.array([0]), atol=128):
                self.image = out1(image=self.image)
            else:
                self.image = out2(image=self.image)


    def dominantColor1Quadraple(self, out1, out2, out3, out4, image=None):
        """
        Odluka izlaza na temelju dominantne boje unutar slike, odluka je unutar jedne od četiri kategorije
        """
        if out1:
            if np.allclose(self.dominantColor1, np.array([32]), atol=32):
                self.image = out1(image=self.image)
            elif np.allclose(self.dominantColor1, np.array([96]), atol=32):
                self.image = out2(image=self.image)
            elif np.allclose(self.dominantColor1, np.array([160]), atol=32):
                self.image = out3(image=self.image)
            elif np.allclose(self.dominantColor1, np.array([224]), atol=32):
                self.image = out4(image=self.image)


    def process(self, original, mask, no_cells, individual):
        """
        Vraća razliku originalne slike i maske nakon trasnfomacija zadanih unutar stabla
        :param original: Originalna slika
        :param mask: Korespondentna maska
        :param individual:Stablasta struktura
        :return:Vrijednost detektirane slike
        """
        self._reset(original, mask)
        cv2.imshow("Prije", self.image)
        tree = gp.PrimitiveTree(individual)
        gp.compile(tree, pset)


        detected_cells = image_process.get_number_of_cells(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY))

        if detected_cells > 2 * no_cells: detected_cells = 0

        det = (detected_cells / no_cells) if detected_cells <= no_cells else 2 - (detected_cells / no_cells)

        self.image = image_process.otsu_treshold(self.image)
        cv2.imshow("threshold", self.image)
        cv2.waitKey()
        return morphology_transformation.compare(self.image, self.mask) * det


iprocessor = ImageProcessor()

pset = gp.PrimitiveSet("MAIN", 0)
pset.addPrimitive(iprocessor.dominantColor2Quadraple, 4)
pset.addPrimitive(iprocessor.dominantColor1Bipolar, 2)
pset.addPrimitive(iprocessor.dominantColor1Quadraple, 4)

for i in range(4):
    for j in range(1, 28):
        pset.addTerminal(
            lambda image: morphology_transformation.defined_transforms[i](
                image,
                kernel=structuring_elements.elements.get(j)
            ), "MT{}{}".format(i, j)
        )

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_init", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluateFunction(individual):
    sum = 0
    for i, (name, info) in enumerate(image_process.IMAGES_WITH_INFO.items()):
        if i == TRAIN_NO:
            break

        image, gray_image, mask, no_cells = info

        sum += iprocessor.process(image, mask, no_cells, individual)

    return (sum / TRAIN_NO),


toolbox.register("evaluate", evaluateFunction)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


# Otkomentirati liniju za paralelizam (parametri: -m scoop)
# toolbox.register("map", futures.map)


def main():
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox,
                        cxpb=CROSS_PROBABILITY,
                        mutpb=MUTATION_PROBABILITY,
                        ngen=NUMB_OF_GENERATIONS,
                        stats=stats, halloffame=hof)

    return pop, hof, stats


if __name__ == "__main__":
    image_process.load('images_serialized/images_with_info.dict')
    pop, hof, stats = main()
    print(str(gp.PrimitiveTree(hof[0])))
