# Information Gain

import numpy as np
#import utility_ig    as ut

# Dispersion entropy
def entropy_disp(data, m, t, c): 

    # Normalizamos la información obtenida
    data_normalizada = norm_data_sigmoidal(data)

    # Calculamos los vectores embebidos
    M = len(data_normalizada) - (m - 1) * t
    embedding_vectors = np.array([data_normalizada[i:i + m * t:t] for i in range(M)])

    # Paso 3: Mapear cada vector-embedding en c-símbolos
    symbols = np.round(c * embedding_vectors + 0.5).astype(int)
    # Paso 4: Convertir el vector de símbolos en un número correspondiente a un patrón
    patterns = np.array([1 + np.dot(symbol - 1, c ** np.arange(m)[::-1]) for symbol in symbols])

    # Paso 5: Contar la frecuencia de ocurrencia de cada patrón
    r = c ** m
    freq = np.zeros(r)
    unique, counts = np.unique(patterns, return_counts=True)
    unique = unique.astype(int)  # Asegurarse de que los índices sean enteros
    freq[unique - 1] = counts  # ajustar índices para que correspondan al rango [0, r-1] -> [1, r]
    
    # Paso 6: Calcular la probabilidad de cada patrón
    p = freq / len(patterns)
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
    entropia_y = entropy_disp(clases, m, t, c)
    
    N, d = data.shape
    print(f"Número de muestras (N): {N}, Número de variables (d): {d}")

    valores_ig = []

    bins = int(np.sqrt(N))
    print(f"Dividiendo cada variable en {bins} bins (categorías).")

    for j in range(d):
        bin_edges = np.linspace(np.min(data[:, j]), np.max(data[:, j]), bins + 1)
        data_binned = np.digitize(data[:, j], bin_edges) - 1

        entropia_condicional = 0
        for b in range(bins):
            bin_indices = (data_binned == b)
            Y_in_bin = clases[bin_indices]
            if len(Y_in_bin) > 0:
                d_ji = len(Y_in_bin)
                bin_entropy = entropy_disp(Y_in_bin, m, t, c)
                entropia_condicional += (d_ji / N) * bin_entropy
                print(f"  Bin {b + 1}/{bins}: DE(Y|X=b) = {bin_entropy}, d_ji = {d_ji}")

        ig_j = entropia_y - entropia_condicional
        valores_ig.append((j, ig_j))
        print(f"IG para la variable {j + 1}/{d}: {ig_j}")

    valores_ig = sorted(valores_ig, key=lambda x: x[1], reverse=True)

    print("\nVariables ordenadas por Ganancia de Información (IG):")
    for idx, ig in valores_ig:
        print(f"  Variable {idx + 1} - IG: {ig}")

    # Paso 4: Seleccionar las top-k variables relevantes
    selected_features_indices = [idx + 1 for idx, _ in valores_ig[:top_k]]  # Ajuste de índice +1 para contar desde 1
    print(f"\nTop-{top_k} variables seleccionadas: {selected_features_indices}")

    # Paso 5: Crear la nueva base de datos con las variables seleccionadas (ajustando a índice base 0 para el slicing)
    X_reduced = data[:, [idx - 1 for idx in selected_features_indices]]

    # Guardar todos los índices ordenados en Idx_variable.csv
    np.savetxt('Idx_variable.csv', selected_features_indices, delimiter=',', fmt='%d')
    # Guardar la base de datos con solo las variables más relevantes en DataIG.csv
    np.savetxt('DataIG.csv', X_reduced, delimiter=',', fmt='%f')

    print("Archivos 'Idx_variable.csv' y 'DataIG.csv' creados con éxito.")

    return selected_features_indices, X_reduced

# Load dataClass 
def load_data(file_path):   
    # Cargar los datos desde el archivo CSV
    data = np.loadtxt(file_path, delimiter=",", dtype=float)
    return data

# Beginning ...
def main():    
    # Carga de datos
    data_original = load_data("DataClass.csv")
    print("Se cargaron los datos del archivo DataClass.csv")

    X = data_original[:, :-1]  # Características (variables explicativas)   
    Y = data_original[:, -1]   # Etiquetas (clases)
    print(Y)
    print("El tamaño de la matriz de embedding_vectors es:", np.array(X).shape)
    print("Se genero X e Y con éxito.")
    
    # Cargar los parámetros de configuración
    m, t, c, top_k = np.loadtxt("config.csv", delimiter=",", dtype=int, max_rows=4)    
    print("Se cargaron los datos de configuración:", m, t, c, top_k)

    # Calcular el Information Gain
    k_indices, x_reduced = inform_gain(X, Y, m, t, c, top_k)

if __name__ == '__main__':   
	 main()

