# Information Gain

import numpy as np
#import utility_ig    as ut

# Dispersion entropy
def entropy_disp(data, m, t, c): 

    # Normalizamos la información obtenida
    data_normalizada = norm_data_sigmoidal(data)

    # Calculamos los vectores embebidos
    M = len(data_normalizada) - (m - 1) * t
    vectores_embebidos = np.array([data_normalizada[i:i + m * t:t] for i in range(M)])

    # Paso 3: Mapear cada vector-embedding en c-símbolos
    y = np.round(c * vectores_embebidos + 0.5).astype(int)
    # Paso 4: Convertir el vector de símbolos en un número correspondiente a un patrón
    patron_k = np.array([1 + np.dot(y_i - 1, c ** np.arange(m)[::-1]) for y_i in y])

    # Paso 5: Contar la frecuencia de ocurrencia de cada patrón
    unique, counts = np.unique(patron_k, return_counts=True)
    unique = unique.astype(int)  # Asegurarse de que los índices sean enteros
    
    # Paso 6: Calcular la probabilidad de cada patrón
    p = counts / M
    p = p[p > 0]  # Eliminar probabilidades cero para evitar log(0)

    DE = -np.sum(p * np.log2(p))

    return DE

# Normalised by use sigmoidal
def norm_data_sigmoidal(x):
    mu_x = np.mean(x)  # Media de la variable x
    sigma_x = np.std(x)  # Desviación estándar de la variable x
    epsilon = 1e-10  # Pequeña constante para evitar divisiones por cero
    # Paso 1: Calcular u
    u = (x - mu_x) / (sigma_x + epsilon)
    # Paso 2: Calcular X normalizado
    X_normalizado = 1 / (1 + np.exp(-u))
    
    return X_normalizado

#Information gain
def inform_gain(data, clases, m, t, c, top_k):    
    entropia_y = entropy_disp(clases, m, t, c) # Entropía de la variable de salida Y
    
    N, d = data.shape

    valores_ig = [] # Vector para almacenar los valores de IG de cada variable

    bins = int(np.sqrt(N)) # Número de bins (categorías) para discretizar las variables

    for j in range(d):
        bin_edges = np.linspace(np.min(data[:, j]), np.max(data[:, j]), bins + 1) # Obtener límite inferior y superior de los bins
        data_binned = np.digitize(data[:, j], bin_edges) - 1 # Indicar a que bin pertenece cada valor

        entropia_condicional_dj = 0

        for b in range(bins):
            bin_indices = (data_binned == b)
            Y_in_bin = clases[bin_indices]
            if len(Y_in_bin) > 0:
                d_ji = len(Y_in_bin)
                bin_entropy = entropy_disp(Y_in_bin, m, t, c)
                entropia_condicional_dj += (d_ji / N) * bin_entropy

        ig_j = entropia_y - entropia_condicional_dj
        valores_ig.append((j, ig_j))

    valores_ig = sorted(valores_ig, key=lambda x: x[1], reverse=True)

    indices_ordenados = [idx + 1 for idx, _ in valores_ig]  # Ajuste de índice +1 para contar desde 1

    # Paso 4: Seleccionar las top-k variables relevantes
    selected_features_indices = [idx + 1 for idx, _ in valores_ig[:top_k]]  # Ajuste de índice +1 para contar desde 1
    print(f"\nTop-{top_k} variables seleccionadas: {selected_features_indices}")

    # Paso 5: Crear la nueva base de datos con las variables seleccionadas (ajustando a índice base 0 para el slicing)
    X_reduced = data[:, [idx - 1 for idx in selected_features_indices]]

    return indices_ordenados, X_reduced

# Load dataClass 
def load_data(file_path):   
    # Cargar los datos desde el archivo CSV
    data = np.genfromtxt(file_path, delimiter=',', dtype=float)
    return data

# Beginning ...
def main():    
    # Carga de datos
    data_original = load_data("DataClass.csv")

    caracteristicas = data_original[:, :-1]  # Características (variables explicativas)   
    clases = data_original[:, -1]   # Etiquetas (clases)
    
    # Cargar los parámetros de configuración
    m, t, c, top_k = np.loadtxt("config.csv", delimiter=",", dtype=int, max_rows=4)    

    # Calcular el Information Gain
    k_indices, x_reduced = inform_gain(caracteristicas, clases, m, t, c, top_k)

    # Guardar todos los índices ordenados en Idx_variable.csv
    np.savetxt('Idx_variable.csv', k_indices, delimiter=',', fmt='%d')
    # Guardar la base de datos con solo las variables más relevantes en DataIG.csv
    np.savetxt('DataIG.csv', x_reduced, delimiter=',', fmt='%f')

if __name__ == '__main__':   
	 main()

