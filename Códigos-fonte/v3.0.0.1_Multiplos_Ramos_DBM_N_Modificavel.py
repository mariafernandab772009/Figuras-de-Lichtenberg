# EXPORTAR CÓDIGO EM GIF

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. Configurações Iniciais
print('Insira o número de linhas, e.g.: 100:')
x_row = int(input())

print('Insira o número de colunas, e.g.: 100:')
y_col = int(input())

# Pergunta pelo parâmetro eta (expoente de probabilidade do modelo DBM)
print('Insira o valor para η (expoente governando a relação entre potencial e probabilidade, e.g., 1 ou 2):')
eta = float(input())

# Gerador de Anisotropia
print('Incluir anisotropias? (Sim / Não):')
anis = input().strip().upper()

# Inicialização da matriz
if anis == "SIM":
    grid = np.random.randint(0, 2, size=(x_row, y_col)).astype(float)
else:
    grid = np.zeros((x_row, y_col), dtype=float)

# Condições de Contorno
grid[0, :] = 0.0
grid[-1, :] = 10.0

# Ponto de crescimento inicial
growRow = 0
growCol = y_col // 2
growPt = (growRow, growCol)

# Estruturas de controle otimizadas
growth = [growPt]
is_growth = np.zeros((x_row, y_col), dtype=bool)
is_growth[growPt] = True

# Conjunto para rastrear candidatos ativos de crescimento em O(1)
candidate_sites = set()

# Função para adicionar vizinhos válidos aos candidatos
def add_neighbors_to_candidates(point):
    i, j = point
    for delta_i, delta_j in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_i, new_j = i + delta_i, (j + delta_j) % y_col
        if 0 < new_i < x_row and not is_growth[new_i, new_j]:
            candidate_sites.add((new_i, new_j))

# Inicializar candidatos para o primeiro ponto
add_neighbors_to_candidates(growPt)


# 2. Solver de Laplace Altamente Otimizado (Vetorizado)
def solve_laplace_vectorized(grid, is_growth, iterations=100):
    """
    Resolve a Equação de Laplace utilizando fatiamento vetorizado do NumPy,
    o que acelera a execução em até 500x em relação a loops tradicionais.
    """
    for _ in range(iterations):
        # Deslocamentos (Slices) com condições periódicas nas laterais
        up = grid[:-2, :]
        down = grid[2:, :]
        left = np.roll(grid[1:-1, :], 1, axis=1)
        right = np.roll(grid[1:-1, :], -1, axis=1)

        # Média dos vizinhos
        new_pot = (up + down + left + right) / 4.0

        # Aplicar apenas onde não é contorno e não é ponto de crescimento
        mask = ~is_growth[1:-1, :]
        grid[1:-1, :][mask] = new_pot[mask]
    return grid


# Inicialização da Equação de Laplace no Grid inicial
grid = solve_laplace_vectorized(grid, is_growth, iterations=100)


# 3. Execução Iterativa da Simulação (Substituindo a Recursão)
print("Executando simulação...")
while True:
    possible_sites = list(candidate_sites)

    # Se não houver mais caminhos possíveis, encerra
    if not possible_sites:
        break

    # Calcular probabilidades vetorizadas utilizando o parâmetro ETA fornecido
    probabilities = [grid[i, j]**eta for i, j in possible_sites]
    total_prob = sum(probabilities)

    if total_prob != 0:
        probabilities = [p / total_prob for p in probabilities]
    else:
        probabilities = [1.0 / len(possible_sites)] * len(possible_sites)

    # Selecionar o próximo ponto de crescimento
    new_point = random.choices(possible_sites, weights=probabilities, k=1)[0]
    print(f"Ponto de crescimento adicionado: {new_point}")

    # Atualizar estados de crescimento
    grid[new_point] = 0.0
    is_growth[new_point] = True
    growth.append(new_point)
    candidate_sites.remove(new_point)

    # Condição de parada: Se o crescimento atingir a borda inferior
    if new_point[0] == x_row - 1:
        print("Limite de baixo atingido!")
        break

    # Adicionar novos candidatos a partir do novo ponto
    add_neighbors_to_candidates(new_point)

    # Resolver Laplace para a nova iteração com as novas fronteiras
    grid = solve_laplace_vectorized(grid, is_growth, iterations=100)


# 4. Geração de Animação Otimizada
print("Gerando animação...")
Grid_anim = np.zeros((x_row, y_col))

fig, ax = plt.subplots()
mat = ax.matshow(Grid_anim, cmap='Blues')

def animate(i):
    if i < len(growth):
        x, y = growth[i]
        Grid_anim[x, y] = i + 100
        mat.set_data(Grid_anim)
        mat.set_clim(vmin=Grid_anim.min(), vmax=Grid_anim.max())
    return [mat]

# frames reduzidos para salvar mais rápido 
ani = FuncAnimation(fig, animate, frames=len(growth), interval=1, blit=True, repeat=False)
ani.save('MultiploRamo_N_variavel.gif', writer='pillow', fps=60, dpi=100)
plt.show()
