import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

#constantes
Kb = 8.6173324e-5 #ev - cibstabte de boltzmann

SIZE = 4 # Ordem da matriz
TEMPERATURAS = [i for i in range(15,600,40)] # Range de temperaturas
#TEMPERATURAS = [30,40,50]

E_A = 0 # Energia on site "A"
E_B = 0 # Energia on site "B"
E_AA = -0.02 # Energia interacao entre "A" e "A"
E_BB = -0.02 # Energia interacao entre "B" e "B"
E_AB = -0.01 # Energia interacao entre "A" e "B"

N_A_range = np.array([n for n in np.arange(0,(SIZE*SIZE)+1,1)]) # Range da quantidade de "A"
N_B_range = N_A_range[::-1] # Range da quantidade de "B" considerando "A"
N_A_percentage = N_A_range/(SIZE*SIZE) # Concentracao

def possible_matrices(N_a, N_b):
    """
    N_a: numero de elementos "A"
    N_b: numero de elementos "B"
    """

    global SIZE
    
    # Check sum
    if N_a + N_b != SIZE * SIZE:
        raise ValueError("The total number of 'A's and 'B's should match the size of the matrix.")
    
    if N_a == 0:
        return np.full((1, SIZE, SIZE), 'B', dtype='<U1')
    
    # Gerar combinacoes
    indices_A = np.array(list(itertools.combinations(range(SIZE*SIZE), N_a)))
    
    # Gerar matrizes full "B"
    matrices = np.full((len(indices_A), SIZE, SIZE), 'B', dtype='<U1')
    
    # Indices possiveis "A"
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

        # Vizinhos à direita e abaixo
        det_vizinhos = lambda x, y: vizinhos[(i, j)].append(matriz[x, y])
        if j < colunas - 1: det_vizinhos(i, j + 1)
        if i < linhas - 1: det_vizinhos(i + 1, j)

        # Vizinhos nas bordas
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

    global E_A, E_B, E_AA, E_BB, E_AB

    vizinhos_dict = determinar_vizinhos(matriz)
    energias_vizinhos = np.array([])

    E_on_site = N_a * E_A + N_b * E_B

    for coordenada, vizinhos in vizinhos_dict.items():
        i, j = coordenada
        elemento = matriz[i, j]
        vizinhos = np.array(vizinhos)

        # Energia entre o elemento e seus vizinhos
        energia_elemento_vizinhos = np.where(elemento != vizinhos, E_AB, 
                                              np.where(elemento == 'A', E_AA, E_BB))
        energias_vizinhos = np.concatenate((energias_vizinhos, energia_elemento_vizinhos))

    energia_célula = np.sum(energias_vizinhos) + E_on_site
    return energia_célula

def helmholtz(energia_matrix_list, temperatura):
    """Essa função calcula o valor da energia de helmholtz com base na função de partição.
    
    Args:
        
        energia_matrix_list: lista com a energia de cada configuração da celula.
        temperatura: temperatura da celula.
        
    Return:
    
        F: energia de helmholtz da concentracao
    """

    global Kb

    beta = 1.0 / (Kb * temperatura)
    Z = 0.0
    for energia in energia_matrix_list:
        Z += np.exp(-beta * energia)

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

    global Kb, E_AA, E_AB, E_BB

    if (sigma == 0) or (sigma == 1):
        return 0

    xi = 4/(Kb*T) * (E_AB - (E_AA + E_BB)/2)
    first = sigma*np.log(sigma)
    second = (1-sigma)*np.log(1-sigma)
    third = xi*sigma*(1-sigma)

    F = Kb*T * (first + second + third)


    return F

def determinar_ensemble_canonico(temperaturas):
    """ Essa funcao determina o DataFrame para o plot da energia de helmholtz em relacao a concentracao com variacao de temperatura, baseado no ensemble canonico.

    Args:
        temperaturas: lista de temperaturas a serem utilizadas.

    Return:
        data: DataFrame contendo as concentracoes discretas, razao de concentracao,energia de hemholtz, temperaturas e a quantidade de particulas "A".

    """

    global SIZE, N_A_range, N_B_range, N_A_percentage, E_AA, E_BB, E_AB

    energia_matrix_list = []
    energia_helmholtz_list = []
    temperaturas_list = []

    for T in tqdm(temperaturas,desc= "Temperature"):
        for N_a, N_b in tqdm(zip(N_A_range,N_B_range), desc= "Concentration"):
            energia_matrix_list = []
            matrices = possible_matrices(N_a, N_b)
            

            for matrix in tqdm(matrices, desc= "Permutation"):
                energia_matrix = energia_total(matrix,N_a,N_b)
                energia_matrix_list.append(energia_matrix)
        
            energia_helmholtz = helmholtz(energia_matrix_list,T)
            energia_helmholtz_list.append(energia_helmholtz)
        
    temperaturas_list = [T for T in temperaturas for _ in range(0,(SIZE*SIZE) + 1)]
    x_axis = list(N_A_range) * len(temperaturas)
    concentration_ratio = list(N_A_percentage) * len(temperaturas)
    Na_Nb = [[i, (SIZE*SIZE) - i] for i in range((SIZE*SIZE) + 1)] * len(temperaturas)

    data = pd.DataFrame({"[N_a, N_b]": Na_Nb, "Ratio": concentration_ratio ,"Energia de Helmholtz [eV]": energia_helmholtz_list, "Temperatura [K]": temperaturas_list, "x_axis": x_axis})
    return data

def determinar_mean_field(temperaturas):
    """ Essa funcao determina o DataFrame para o plot da energia de helmholtz em relacao a concentracao com variacao de temperatura, baseado na aproximacao de campo medio.

    Args:
        temperaturas: lista de temperaturas a serem utilizadas.

    Return:
        data: DataFrame contendo as concentracoes discretas, razao de concentracao,energia de hemholtz, temperaturas e a quantidade de particulas "A".

    """

    global N_A_percentage, SIZE

    F_list = []
    

    for T, percentage in itertools.product(temperaturas, N_A_percentage):
        F = pappu(T, percentage)
        F_list.append(F)

    temperaturas_list = [T for T in temperaturas for _ in range(0,(SIZE*SIZE) + 1)]
    x_axis = list(N_A_range) * len(temperaturas)
    concentration_ratio = list(N_A_percentage) * len(temperaturas)
    Na_Nb = [[i, (SIZE*SIZE) - i] for i in range((SIZE*SIZE) + 1)] * len(temperaturas)

    data = pd.DataFrame({"[N_a, N_b]": Na_Nb,"Ratio": concentration_ratio, "Energia de Helmholtz [eV]": F_list, "Temperatura [K]": temperaturas_list, "x_axis": x_axis})
    return data

def plot_helmholtz(data, img_name, color_pallete='tab10'):
    """ Essa funcao plota algum dos dataframes.

    Args:
        data: DataFrame contendo os dados.
        img_name: Nome em string da figura a ser salva. Deve conter o formato a ser salvo.
        color_palette: palheta de cores usada para o plot.


    Return:
        data: DataFrame contendo as concentracoes discretas, razao de concentracao,energia de hemholtz, temperaturas e a quantidade de particulas "A".

    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=data["x_axis"], y=data["Energia de Helmholtz [eV]"],hue=data["Temperatura [K]"],palette=color_pallete)
    plt.xlim(0, None)
    #plt.ylim(0, None)
    plt.xticks(data["x_axis"])
    plt.title('Concentração x Temperatura')
    plt.xlabel('Concentração de "A"')
    plt.ylabel('Energia de Hemlholtz [eV]')
    plt.tight_layout()
    fig.get_figure() #'pega' o gráfico para salvar
    fig.savefig(img_name) #salva o gráfico 

ensemble_canonico = determinar_ensemble_canonico(TEMPERATURAS)

mean_field = determinar_mean_field(TEMPERATURAS)

plot_helmholtz(ensemble_canonico, img_name = "ensemble_canonico.png")

plot_helmholtz(mean_field, img_name = "mean_field.png")