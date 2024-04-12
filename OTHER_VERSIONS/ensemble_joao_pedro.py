# Plot heatmap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# Grid permutations
from sympy.utilities.iterables import multiset_permutations
# Plot
import matplotlib.pyplot as plt
import matplotlib as mpl
# TQDM
from tqdm import tqdm, trange
# Random choices
from random import choice
# Numpy
import numpy as np



# Constants
# Boltzmann
Kb = 1#.380649e-23
# Counted neighbours
neighbours = [(0, -1), (0, 1), (-1, 0), (1, 0)]



# Classes
# Class to facilitate energy calculations
class particleSolvent():
    sideDecoder = {
        (0, -1): 'left',
        (0, 1): 'right',
        (-1, 0): 'down',
        (1, 0): 'upper',
    }
    invertedNeighbour = {
        'left': 'right',
        'right': 'left',
        'upper': 'down',
        'down': 'upper'
    }

    def __init__(self, particleSolventType) -> None:
        self.type = particleSolventType
        self.neighbours = {
            'left': 0,
            'right': 0,
            'upper': 0,
            'down': 0
        }

        return
    
    def getParticleSolventType(self):
        return self.type

    def getParticleSolventNeighbourState(self, neighbourSide):
        return self.neighbours[neighbourSide]

    def getEnergyFromNeighbour(self, neighbour, neighbourSide, checkReciprocal, selfState=1):
        global interactionEnergies

        neighbourSide = self.sideDecoder[neighbourSide]
        if neighbour.getParticleSolventNeighbourState(neighbourSide) and checkReciprocal:
            return 0

        self.changeSelfState(neighbourSide, selfState)
        neighbour.changeNeighbourState(neighbourSide, selfState)

        neighbourType = neighbour.getParticleSolventType()
        energy = interactionEnergies[f'E{self.type[0]}{neighbourType[0]}']

        return energy
    
    def changeSelfState(self, neighbourSide, state=1):
        self.neighbours[self.invertedNeighbour[neighbourSide]] = state

        return True

    def changeNeighbourState(self, neighbourSide, state=1):
        self.neighbours[neighbourSide] = state

        return True



# Functions
def calculateFractionEnergiesPlot(configs, plotConfigs, pbars=None):
    # Global variables
    global Kb, gridSize, globalAdjust

    particleFractions = configs['particleFractions']
    temperature = configs['temperature']

    axHelmholtz = plotConfigs['axHelmholtz']
    axMean = plotConfigs['axMean']
    colorsHot = plotConfigs['colorsHot']
    colorsTurbo = plotConfigs['colorsTurbo']
    line = plotConfigs['lineStyle']
    marker = plotConfigs['marker']
    size = plotConfigs['size']
    labelPlot = plotConfigs['labelPlot']

    # All calculated Helmholtz energies
    allHelmholtz = []

    if pbars == None:
        pbar1 = tqdm(particleFractions, total=len(particleFractions))
        pbar2 = tqdm()
    else:
        pbar1 = pbars['pbar1']
        pbar2 = pbars['pbar2']
        pbar1.reset(len(particleFractions))

    # Iterating over all fractions of particles
    for numPa, numPb in particleFractions:
        pbar1.set_description(f'Fraction = ({numPa}, {numPb})')

        # Setting the initial permutation configuration
        initialConfiguration = ['A' for _ in range(numPa)] + ['B' for _ in range(numPb)]

        # Calculated energies for this fraction
        helmholtzEnergies = []

        permutations = list(multiset_permutations(initialConfiguration))
        pbar2.reset(total=len(permutations))
        pbar2.set_description(f'Permutations')

        # Iterating over all permutations
        for permutation in permutations:
            # Setting the permutation to be a matrix
            permutation = [particleSolvent('particle') if iteration == 'A' else particleSolvent('solvent') for iteration in permutation]
            matrix = np.reshape(np.array(permutation), (-1, gridSize))
            # Setting the iterator for looking each individual particle
            iterator = np.nditer(matrix, flags=['multi_index', 'refs_ok'])
            # Calculated energies for this permutation
            permutationEnergy = 0

            # Iterating over each particle to check for neighbours
            for particle in iterator:
                # Setting the particle to the actual object
                particle = particle.item()
                # Index of current particle
                particleIndex = iterator.multi_index

                # Iterating over each of the particles neighbour
                for neighbourIndex in neighbours:
                    # Logic for cheking neighbour position
                    neighbourCheck = []
                    checkReciprocal = False
                    for i, j in zip(particleIndex, neighbourIndex):
                        # Getting possibility of reciprocal
                        if i+j != gridSize:
                            neighbourCheck.append(i+j)
                        else:
                            checkReciprocal = True
                            neighbourCheck.append(0)
                    # Getting the values for the neighbour and the particle
                    neighbour = matrix[*neighbourCheck]
                    # Getting the energy for their interaction
                    energy = particle.getEnergyFromNeighbour(neighbour, neighbourIndex, checkReciprocal)
                    # Summing to the energy of this permutation
                    permutationEnergy += energy

            # Adding this permutation to be calculated on the overall energy (Z)
            helmholtzEnergies.append(permutationEnergy)

            pbar2.update()
        pbar2.refresh()
        pbar1.update()
        
        z = 0
        for Ei in helmholtzEnergies:
            z += np.exp( (-Ei) / (temperature*Kb) )
        
        allHelmholtz.append(z)

    pbar1.refresh()

    xAxisHelmholtz = [a/(len(particleFractions)-1) for a, b in particleFractions]
    xAxisMean = [a/(len(particleFractions)-1) for a, b in particleFractions]

    allHelmholtz = [-Kb*temperature * np.log(z)/particleFractions[0][1] +globalAdjust for z in allHelmholtz]
    meanField = [pappu(temperature, sigma) for sigma in xAxisHelmholtz]

    axHelmholtz.plot(xAxisHelmholtz, allHelmholtz, label=labelPlot, marker=marker, linestyle=line, color=colorsHot, markersize=size)
    axMean.plot(xAxisMean, meanField, label=labelPlot, marker=marker, linestyle=line, color=colorsTurbo, markersize=size)

def generateAX(subplots=(1, 1), figsize=(5, 5), width_ratios=None):
    fig, ax = plt.subplots(subplots[0], subplots[1], figsize=figsize, dpi=200, sharey='row', width_ratios=width_ratios)
    return fig, ax

def everything(configs, plotConfigs, pbars=None):
    globalPbar = trange(len(configs), total=len(configs))
    if pbars == None:
        pbars = {
            'pbar1': tqdm(),
            'pbar2': tqdm()
        }

    for i in globalPbar:
        globalPbar.set_description(f"Temperature = {configs[i]['temperature']} eV")
        calculateFractionEnergiesPlot(configs[i], plotConfigs[i], pbars=pbars)

def colorCodeGenerator(number):
    if number <= 2:
        return ['#9A32CD', '#00F5FF']

    colors = []
    for _ in range(number):
        a = [choice('0123456789ABCDEF') for _ in range(6)]
        colors.append('#' + ''.join(a))
    return colors

def pappu(T, sigma):
    global Kb, interactionEnergies

    if (sigma == 0) or (sigma == 1):
        return 0

    xi = 4/(Kb*T) * (interactionEnergies['Eps'] - (interactionEnergies['Epp'] + interactionEnergies['Ess'])/2)
    first = sigma*np.log(sigma)
    second = (1-sigma)*np.log(1-sigma)
    third = xi*sigma*(1-sigma)

    F = Kb*T * (first + second + third)

    # print(sigma)
    # print(first, second, third, xi)
    # print(F)
    return F


# Problem variables
# Plot temperature
# temperature = .026 # eV #300 # K
# Size of the grid
gridSize = 4
# Number of atoms in the grid
particleNumber = gridSize**2
# Number of particles of each type
# particleFractions = [(x, particleNumber-x) for x in range(particleNumber+1)]

# Variables debug
# print(particleFractions)



# Test energies
# Energy particle-particle
Epp = -.02
# Energy solvent-solvent
Ess = -.02
# Energy particle-solvent
Eps = -.005
# Dictionary for easier value recognition
interactionEnergies = {'Epp': Epp, 'Ess': Ess, 'Eps': Eps, 'Esp': Eps}



tempsMinMax = [.0065, .065]
globalParticleFractions = [(x, particleNumber-x) for x in range(particleNumber+1)]
globalTemperatures = np.linspace(tempsMinMax[0], tempsMinMax[1], num=15) #[.0065*(i+1) for i in range(3)]

suffix = 'Turbo'
# colors = colorCodeGenerator(len(globalTemperatures))
# colors = ['#9432CD', '#C23030', '#0317FC', '#FC03E3', '#0FFC03']
cmapHot = mpl.colormaps[suffix.lower()]
cmapTurbo = mpl.colormaps[suffix.lower()]
colorsHot = [cmapHot(i/len(globalTemperatures)) for i in range(1+len(globalTemperatures))]
colorsTurbo = [cmapTurbo(i/len(globalTemperatures)) for i in range(1+len(globalTemperatures))]

configs = {
    i: {
        'particleFractions': globalParticleFractions,
        'temperature': globalTemperatures[i]
    } 
    for i in range(len(globalTemperatures))
}

fig, (axHelmholtz, axMean, axMerda) = generateAX(subplots=(1, 3), figsize=(7, 3), width_ratios=[1, 1, .3])
plotConfigs = {
    i: {
        'axHelmholtz': axHelmholtz,
        'axMean': axMean,
        'colorsHot': colorsHot[i],
        'colorsTurbo': colorsTurbo[i],
        'lineStyle': '-',
        'marker': '',
        'size': 6,
        'labelPlot': f"{configs[i]['temperature']:.3} eV"
    }
    for i in range(len(configs))
}

globalAdjust = .07



# Running the problem
everything(configs, plotConfigs)

for ax in [axHelmholtz, axMean]:
    ax.set_xlabel('Fração do particulado')
    # ax.legend(fontsize=4)
axHelmholtz.set_ylabel('Energia livre de Helmholtz (u.a.)')


orientation, label = 'vertical', 'T (eV)'
scalarHot = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(*tempsMinMax), cmap=cmapHot)
colorBarAx = fig.add_axes([.85, .195, 0.03, 0.755])
cb = plt.colorbar(scalarHot, cax=colorBarAx, orientation=orientation, ticks=tempsMinMax, label=label, pad=.01)
axMerda.set_axis_off()

axHelmholtz.set_title('Abordagem computacional extensiva', fontsize=10)
axMean.set_title('Abordagem de campo médio', fontsize=10)

fig.tight_layout()
fig.savefig(f'plot{suffix}.png', format='png')

plt.show()