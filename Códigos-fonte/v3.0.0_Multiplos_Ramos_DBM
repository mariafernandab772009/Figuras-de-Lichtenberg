# EXPORTAR CÓDIGO EM GIF

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import sys
print(sys.getrecursionlimit())
sys.setrecursionlimit(2000)
print(sys.getrecursionlimit())

# Cria uma matriz retangular de zeros
print('Insira o número de linhas, e.g.: 100:')
x_row = int(input())

print('Insira o número de colunas, e.g.: 100:')
y_col = int(input())

# Gerador de anisotropia
print('Incluir anisotropias? (Sim / Não):')
anis = str(input()).upper()

if anis == "SIM":
    grid = np.random.randint(0, 2, size=(x_row, y_col))

else:
    grid = np.zeros((x_row, y_col))

# Configura condições de limite
grid[0, :] = 0
grid[-1, :] = 10

# Cria uma lista de pontos de crescimento
growth = []

# Configura o sítio de crescimento
'''
print('Enter a starting position, e.g.: 0, 0:')
growRow, growCol = input().split(",")
growRow = int(growRow)
growCol = int(growCol)
growPt = (growRow, growCol)
grid[growPt] = 0
growth.append(growPt)
'''
growRow = 0
growCol = y_col // 2
growPt = (growRow, growCol)
growth.append(growPt)

# Inicializa a Equação de Laplace na matriz
iterations = 100
for _ in range(iterations):
    for i in range(1, x_row-1): # Mantém fixos o primeiro e último valores de linha
        for j in range(0, y_col):
            up = grid[i-1, j]
            down = grid[i+1, j]
            left = grid[i, (j-1) % y_col]
            right = grid[i, (j+1) % y_col]
            grid[i, j] = (left + right + up + down) / 4
# Imprime a matriz

# Define o operador de Laplace
def laplaceOperator(grid):
    x_row = len(grid)
    y_col = len(grid[0])
    newGrid = grid.copy()

    for i in range(1, x_row - 1): # Mantém fixos o primeiro e último valores de linha
        for j in range(0, y_col):
            if (i, j) not in growth:
                up = grid[i-1, j]
                down = grid[i+1, j]
                left = grid[i, (j-1) % y_col]
                right = grid[i, (j+1) % y_col]
                newGrid[i, j] = (left + right + up + down) / 4
    return newGrid

# Define a Equação de Laplace
def laplaceEquation(grid, iterations = 100):
    for _ in range(iterations):
        grid = laplaceOperator(grid)
    # Imprime a matriz
    return grid

def simulation(grid):
    global growth
    x_row = len(grid)
    y_col = len(grid[0])
   
    # Encontra todos os possíveis sítios de crescimento
    possibleSites = []
    for i, j in growth:
        for delta_i, delta_j in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Displacements: up, down, left, right
            new_i, new_j = i + delta_i, (j + delta_j) % y_col

            if 0 < new_i < x_row and (new_i, new_j) not in growth:
                possibleSites.append((new_i, new_j))

    # Termina se o algoritmo está preso
    if len(possibleSites) == 0:
        return grid

    # Calcula a probabilidade de crescimento para cada possível sítio de crescimento
    probabilities = []
    for i, j in possibleSites:
        prob = grid[i, j]**2 # Esse é o parâmetro η que poderá ser modificado na próxima versçao
        probabilities.append(prob)

    # Normaliza as probabilidades
    totalProb = sum(probabilities)
    if totalProb != 0:
        probabilities = [p / totalProb for p in probabilities]
    else:
        probabilities = [1 / len(possibleSites)] * len(possibleSites)

    # Seleciona o próximo sítio de crescimento
    newPoint = random.choices(possibleSites, weights = probabilities, k = 1)[0]
    print(newPoint)

    # Adiciona um novo sítio de crescimento à lista de crescimento
    grid[newPoint] = 0
    growth.append(newPoint)

    # Termina se atingiu a base
    if newPoint[0] == x_row - 1:
        return grid

    # Atualiza a matriz e chama recursivamente a função de simulação
    grid = laplaceEquation(grid)
    grid = simulation(grid)

    return grid

# Inicia a simulação
grid = simulation(grid)

# Resultados
print(grid)
print(growth)

# Resultados mais limpos
Grid = np.zeros((x_row, y_col))

# Animação
def animate(i):
    try:
        # Imprime i 
        x = growth[i][0]
        y = growth[i][1]
        Grid[x][y] = i + 100
        ax.clear()
        ax.matshow(Grid, cmap='Blues')

    except IndexError:
        print("Done!")
        return Grid

fig, ax = plt.subplots()

ani = FuncAnimation(fig, animate, frames = len(growth), interval = 0.0001, repeat = False)
ani.save('DBM.gif', writer='pillow', fps=120, dpi=100)
plt.show()

'''
# Simple plotting
newGrid = np.zeros((x_row, y_col))
for i, j in growth:
    newGrid[i, j] = 1

fig, ax = plt.subplots()
ax.matshow(grid, cmap='Blues')
ax.matshow(newGrid, cmap='Blues')
# Ticks
ax.set_xticks(np.arange(-0.5, x_row, 1), minor=True)
ax.set_yticks(np.arange(-0.5, y_col, 1), minor=True)
ax.grid(which='minor', linestyle='-', linewidth=1)

plt.show()
'''
# SALVA A ANIMAÇÃO 
print("Salvando animação como MultiploRamoAnisotropia.gif... Aguarde.")
ani.save('MultiploRamoAnisotropia.gif', writer='pillow', fps=15)
print("Animação salva com sucesso!")
