# WordMetrics.py
import numpy as np

# For some variants, look here https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python

# Pure python
def edit_distance_python2(a, b):
    # Esta versão é comutativa, então, como uma otimização, forçamos |a|>=|b|
    if len(a) < len(b):
        return edit_distance_python(b, a)
    if len(b) == 0:  # Pode lidar com sequências vazias mais rápido
        return len(a)
    # Apenas duas linhas são realmente necessárias: a que está sendo preenchida e a anterior
    distances = []
    distances.append([i for i in range(len(b)+1)])
    distances.append([0 for _ in range(len(b)+1)])
    # Podemos pré-preencher a primeira linha:
    costs = [0 for _ in range(3)]
    for i, a_token in enumerate(a, start=1):
        distances[1][0] += 1  # Lida com a primeira coluna.
        for j, b_token in enumerate(b, start=1):
            costs[0] = distances[1][j-1] + 1
            costs[1] = distances[0][j] + 1
            costs[2] = distances[0][j-1] + (0 if a_token == b_token else 1)
            distances[1][j] = min(costs)
        # Mover para a próxima linha:
        distances[0][:] = distances[1][:]
    return distances[1][len(b)]

# https://stackabuse.com/levenshtein-distance-and-text-similarity-in-python/
def edit_distance_python(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])
