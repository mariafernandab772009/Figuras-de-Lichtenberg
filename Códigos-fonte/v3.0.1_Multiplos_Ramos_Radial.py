# EXPORTAR CÓDIGO EM GIF

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------------------------------------------------
# 1. Configurações e Entradas do Usuário
# ---------------------------------------------------------
print('Insira o raio, e.g.: 50:')
radius = int(input())
N = 2 * radius + 1
r = (N - 1) // 2
x0, y0 = N // 2, N // 2  # Centro da grade

print('Insira o valor de η (expoente governando a relação entre potencial e probabilidade, e.g., 1 ou 2):')
eta = float(input())

print('Incluir anisotropias? (Sim / Não):')
anis = str(input()).strip().upper()

# Inicialização da grade
if anis == "SIM":
    grid = np.random.randint(0, 2, size=(N, N)).astype(float)
else:
    grid = np.zeros((N, N), dtype=float)

# ---------------------------------------------------------
# 2. Midpoint Circle Algorithm (Construção da Borda)
# ---------------------------------------------------------
perimeter = []
f = 1 - r
dx, dy = 1, -2 * r
x, y = 0, r

perimeter.extend([
    (x0, y0 + r), (x0, y0 - r),
    (x0 + r, y0), (x0 - r, y0)
])

while x < y:
    if f >= 0:
        y -= 1
        dy += 2
        f += dy
    x += 1
    dx += 2
    f += dx

    perimeter.extend([
        (x0 + x, y0 + y), (x0 - x, y0 + y),
        (x0 + x, y0 - y), (x0 - x, y0 - y),
        (x0 + y, y0 + x), (x0 - y, y0 + x),
        (x0 + y, y0 - x), (x0 - y, y0 - x)
    ])

perimeter = list(set(perimeter))  # Remover duplicatas

# Máscaras booleanas de alta performance
is_boundary = np.zeros((N, N), dtype=bool)
for i, j in perimeter:
    is_boundary[i, j] = True

Y, X = np.ogrid[:N, :N]
dist_sq = (X - x0)**2 + (Y - y0)**2

is_interior = (dist_sq < r**2) & (~is_boundary)
is_exterior = (dist_sq > r**2) & (~is_boundary)

# Aplicação das Condições de Contorno
grid[is_boundary] = 1.0
grid[is_exterior] = 0.0

# Ponto Inicial de Crescimento (Centro)
growPt = (x0, y0)
grid[growPt] = 0.0
growth = [growPt]

# Controle O(1)
is_growth = np.zeros((N, N), dtype=bool)
is_growth[growPt] = True

candidate_sites = set()

def add_neighbors_to_candidates(pt):
    i, j = pt
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < N and 0 <= nj < N:
            if is_interior[ni, nj] and not is_growth[ni, nj]:
                candidate_sites.add((ni, nj))

add_neighbors_to_candidates(growPt)

# ---------------------------------------------------------
# 3. Solucionador de Laplace Vetorizado
# ---------------------------------------------------------
update_mask = is_interior & (~is_growth)

def solve_laplace_circular_vectorized(grid, update_mask, iterations=100):
    for _ in range(iterations):
        up = np.roll(grid, 1, axis=0)
        down = np.roll(grid, -1, axis=0)
        left = np.roll(grid, 1, axis=1)
        right = np.roll(grid, -1, axis=1)

        new_pot = (up + down + left + right) / 4.0
        grid[update_mask] = new_pot[update_mask]

    return grid

print("Calculando o potencial inicial de Laplace do campo...")
grid = solve_laplace_circular_vectorized(grid, update_mask, iterations=100)

# ---------------------------------------------------------
# 4. Loop de Simulação DBM
# ---------------------------------------------------------
print("Executando simulação de DBM radial...")
step = 0

while True:
    possible_sites = list(candidate_sites)

    if not possible_sites:
        break

    potentials = np.array([grid[i, j] for i, j in possible_sites])
    potentials = np.maximum(potentials, 0.0)

    probs = potentials ** eta
    total_p = np.sum(probs)

    if total_p > 0:
        probs = probs / total_p
    else:
        probs = np.ones(len(possible_sites)) / len(possible_sites)

    chosen_idx = np.random.choice(len(possible_sites), p=probs)
    new_pt = possible_sites[chosen_idx]

    grid[new_pt] = 0.0
    is_growth[new_pt] = True
    growth.append(new_pt)
    candidate_sites.remove(new_pt)
    update_mask[new_pt] = False

    # Checagem de parada: Tocou na circunferência
    reached_boundary = False
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = new_pt[0] + di, new_pt[1] + dj
        if 0 <= ni < N and 0 <= nj < N:
            if is_boundary[ni, nj]:
                reached_boundary = True
                break

    if reached_boundary:
        print(f"Padrão de descarga formado encostou na borda no passo {step}!")
        break

    add_neighbors_to_candidates(new_pt)
    grid = solve_laplace_circular_vectorized(grid, update_mask, iterations=100)
    step += 1

# ---------------------------------------------------------
# 5. Animação Otimizada com Borda Visível e Exportação
# ---------------------------------------------------------
print("Gerando animação em GIF...")
Grid_anim = np.zeros((N, N))

fig, ax = plt.subplots(figsize=(6, 6))

# Valor alto para garantir que a borda fique azul-escura
BORDER_COLOR_VALUE = len(growth) + 200
Grid_anim[is_boundary] = BORDER_COLOR_VALUE

mat = ax.matshow(Grid_anim, cmap='Blues')

def animate(frame):
    x, y = growth[frame]
    # O gradiente de cor avança conforme a árvore cresce
    Grid_anim[x, y] = frame + 100

    mat.set_data(Grid_anim)
    mat.set_clim(vmin=0, vmax=BORDER_COLOR_VALUE)
    return [mat]

ani = FuncAnimation(
    fig,
    animate,
    frames=len(growth),
    interval=15,
    blit=True,
    repeat=False
)

# Exporta em formato GIF
ani.save('Multiplos_Ramos_Radial.gif', writer='pillow', fps=30, dpi=100)
plt.show()
print("Animação salva com sucesso como 'Multiplos_Ramos_Radial.gif'.")
