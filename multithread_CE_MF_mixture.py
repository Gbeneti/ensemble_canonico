import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# Constants
Kb = 8.6173324e-5  # ev - Boltzmann constant
SIZE = 4  # Matrix size
TEMPERATURAS = np.arange(15, 600, 40)  # Temperature range
E_A = 0  # On-site energy "A"
E_B = 0  # On-site energy "B"
E_AA = -0.02  # Interaction energy "A"-"A"
E_BB = -0.02  # Interaction energy "B"-"B"
E_AB = -0.01  # Interaction energy "A"-"B"
N_A_range = np.arange(0, (SIZE * SIZE) + 1, 1)  # Range of "A" count
N_B_range = N_A_range[::-1]  # Range of "B" count
N_A_percentage = N_A_range / (SIZE * SIZE)  # Concentration

@lru_cache(maxsize=None)
def possible_matrices(N_a, N_b):
    """
    Função que calcula as possíveis matrizes do ensemble.

    Args:
        N_a: numero de elementos "A"
        N_b: numero de elementos "B"
    
    Return:
        matrices: array de matrizes
    """
    if N_a + N_b != SIZE * SIZE:
        raise ValueError("The total number of 'A's and 'B's should match the size of the matrix.")

    if N_a == 0:
        return np.full((1, SIZE, SIZE), 'B', dtype='<U1')
    
    indices_A = np.array(list(itertools.combinations(range(SIZE * SIZE), N_a)))
    matrices = np.full((len(indices_A), SIZE, SIZE), 'B', dtype='<U1')
    
    for i, indices in enumerate(indices_A):
        indices = np.unravel_index(indices, (SIZE, SIZE))
        matrices[i][indices] = 'A'
    
    return matrices

def determinar_vizinhos(matriz):
    """ Essa função fornece um dicionário que relaciona todos os elementos na célula
    a seus vizinhos, tanto os de dentro da célula quanto os de células vizinhas iguais.
    
    Args:
    
        matriz: a célula da qual se quer saber os elementos e seus vizinhos
        
    Return:
    
        vizinhos: um dicionário com as chaves sendo as coordenadas da matriz e os valores os vizinhos dessa coordenada"""
    vizinhos = {}
    linhas, colunas = matriz.shape

    for i, j in itertools.product(range(linhas), range(colunas)):
        vizinhos[(i, j)] = []

        det_vizinhos = lambda x, y: vizinhos[(i, j)].append(matriz[x, y])
        if j < colunas - 1: det_vizinhos(i, j + 1)
        if i < linhas - 1: det_vizinhos(i + 1, j)
        if j == 0: det_vizinhos(i, colunas - 1)
        if j == colunas - 1: det_vizinhos(i, 0)
        if i == 0: det_vizinhos(linhas - 1, j)
        if i == linhas - 1: det_vizinhos(0, j)

    return vizinhos

def energia_total(matriz, N_a, N_b):
    """ Esta função calcula a energia a energia total da celula.
    
    Args:
        matriz: configuração da célula.
        N_a: numero de particulas "A"
        N_b: numero de particulas "B"
        
    Return:
        energia_célula: energia total da configuração da célula.
    """
    vizinhos_dict = determinar_vizinhos(matriz)
    energias_vizinhos = np.array([])

    E_on_site = N_a * E_A + N_b * E_B

    for coordenada, vizinhos in vizinhos_dict.items():
        i, j = coordenada
        elemento = matriz[i, j]
        vizinhos = np.array(vizinhos)

        energia_elemento_vizinhos = np.where(elemento != vizinhos, E_AB, 
                                              np.where(elemento == 'A', E_AA, E_BB))
        energias_vizinhos = np.concatenate((energias_vizinhos, energia_elemento_vizinhos))

    energia_celula = np.sum(energias_vizinhos) + E_on_site
    return energia_celula

def helmholtz(energia_matrix_list, temperatura):
    """Essa função calcula o valor da energia de helmholtz com base na função de partição.
    
    Args:
        
        energia_matrix_list: lista com a energia de cada configuração da celula.
        temperatura: temperatura da celula.
        
    Return:
    
        F: energia de helmholtz da concentracao
    """
    beta = 1.0 / (Kb * temperatura)
    Z = np.sum(np.exp(-beta * np.array(energia_matrix_list)))
    F = -(Kb * temperatura) * np.log(Z)
    return F

def pappu(T, sigma):
    """ Essa função calcula o campo medio.

    Args:
        T: Temperatura em K
        sigma: Concentracao atual de A em relacao ao total.

    Return:
        F: Energia de Helmholtz dada a aproximacao de campo medio.

    """
    if sigma == 0 or sigma == 1:
        return 0

    xi = 4 / (Kb * T) * (E_AB - (E_AA + E_BB) / 2)
    F = Kb * T * (sigma * np.log(sigma) + (1 - sigma) * np.log(1 - sigma) + xi * sigma * (1 - sigma))
    return F

def calcular_energia_helmholtz(N_a, N_b, T):
    """ Calcula a energia de helmholtz para todas as matrizes para uma dada temperatura

    Args:
        N_a: numero de particulas "A"
        N_b: numero de particulas "B"
        T: Temperatura

    Return:
        Energia das matrizes para uma dada temperatura.
    """
    matrices = possible_matrices(N_a, N_b)
    energia_matrix_list = [energia_total(matrix, N_a, N_b) for matrix in matrices]
    return helmholtz(energia_matrix_list, T)

def determinar_ensemble_canonico(temperaturas):
    """ Essa funcao determina o DataFrame para o plot da energia de helmholtz em relacao a concentracao com variacao de temperatura, baseado no ensemble canonico.

    Args:
        temperaturas: lista de temperaturas a serem utilizadas.

    Return:
        data: DataFrame contendo as concentracoes discretas, razao de concentracao,energia de hemholtz, temperaturas e a quantidade de particulas "A".

    """
    results = []
    
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(calcular_energia_helmholtz, N_a, N_b, T): (N_a, N_b, T) 
                   for T in temperaturas for N_a, N_b in zip(N_A_range, N_B_range)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating Helmholtz energy"):
            N_a, N_b, T = futures[future]
            energia_helmholtz = future.result()
            results.append([N_a, N_b, N_a / (SIZE * SIZE), energia_helmholtz, T])

    data = pd.DataFrame(results, columns=["N_a", "N_b", "Concentration Ratio", "Energia de Helmholtz [eV]", "Temperatura [K]"])
    return data

def determinar_mean_field(temperaturas):
    """ Essa funcao determina o DataFrame para o plot da energia de helmholtz em relacao a concentracao com variacao de temperatura, baseado na aproximacao de campo medio.

    Args:
        temperaturas: lista de temperaturas a serem utilizadas.

    Return:
        data: DataFrame contendo as concentracoes discretas, razao de concentracao,energia de hemholtz, temperaturas e a quantidade de particulas "A".

    """
    results = []
    
    for T, percentage in tqdm(itertools.product(temperaturas, N_A_percentage), total=len(temperaturas) * len(N_A_percentage), desc="Calculating mean field"):
        F = pappu(T, percentage)
        results.append([percentage * SIZE * SIZE, (1 - percentage) * SIZE * SIZE, percentage, F, T])
    
    data = pd.DataFrame(results, columns=["N_a", "N_b", "Concentration Ratio", "Energia de Helmholtz [eV]", "Temperatura [K]"])
    return data

def plot_helmholtz(data, img_name, num_particles_discrete=False, color_pallete='tab10'):
    """ Essa funcao plota algum dos dataframes.

    Args:
        data: DataFrame contendo os dados.
        num_particules_discrete: True to make number of particles discrete. False for the particle concentration in %.
        img_name: Nome em string da figura a ser salva. Deve conter o formato a ser salvo.
        color_palette: palheta de cores usada para o plot.


    Return:
        data: DataFrame contendo as concentracoes discretas, razao de concentracao,energia de hemholtz, temperaturas e a quantidade de particulas "A".

    """

    x_axis = "N_a" if num_particles_discrete else "Concentration Ratio"
    img_type = "_discrete" if num_particles_discrete else "_ratio"
    img_name_type = img_name + img_type + ".png"

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=x_axis, y="Energia de Helmholtz [eV]", hue="Temperatura [K]", palette=color_pallete, data=data)
    plt.xlim(0, None)
    plt.xticks(data[x_axis])
    plt.title('Concentração x Temperatura')
    plt.xlabel('Concentração de "A"')
    plt.ylabel('Energia de Helmholtz [eV]')
    plt.tight_layout()
    fig.savefig(img_name_type)

ensemble_canonico = determinar_ensemble_canonico(TEMPERATURAS)
mean_field = determinar_mean_field(TEMPERATURAS)

plot_helmholtz(ensemble_canonico, num_particles_discrete=False, img_name="ensemble_canonico")
plot_helmholtz(mean_field, num_particles_discrete=False, img_name="mean_field")