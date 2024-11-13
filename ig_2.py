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
    nDE = 1/np.log2(c**m) * DE
    return nDE

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

# Función principal para calcular la ganancia de información
def inform_gain(data, clases, m, tau, c, top_k):
    """
    Calcula la ganancia de información para cada variable en 'data'.
    
    Parámetros:
    data (np.array): Matriz de datos donde cada columna es una variable.
    clases (np.array): Vector de clases para cada observación en 'data'.
    m, tau, c (int): Parámetros para el cálculo de entropía de dispersión.
    top_k (int): Número de variables con mayor ganancia de información a seleccionar.
    
    Retorna:
    list: Índices de las variables ordenadas por ganancia de información.
    np.array: Nueva base de datos reducida con las 'top_k' variables.
    """
    
    # Entropía de la variable de salida (clases)
    entropia_y = entropy_disp(clases, m, tau, c)
    
    N, d = data.shape
    valores_ig = []  # Vector para almacenar los valores de IG de cada variable

    # Número de bins para discretizar cada variable
    bins = int(np.sqrt(N))
    
    # Cálculo de la entropía condicional para cada variable en 'data'
    for j in range(d):
        bin_edges = np.linspace(np.min(data[:, j]), np.max(data[:, j]), bins + 1)
        data_binned = np.digitize(data[:, j], bin_edges) - 1

        entropia_condicional_dj = 0

        # Calcular la entropía de dispersión dentro de cada bin
        for b in range(bins):
            # Filtrar las clases correspondientes al bin actual
            Y_in_bin = clases[data_binned == b]
            
            # Solo calcular si hay datos en el bin
            if len(Y_in_bin) > 0:
                # Calcular la entropía de dispersión en el bin actual
                bin_entropy = entropy_disp(Y_in_bin, m, tau, c)
                
                # Ponderar la entropía del bin por su proporción en el total
                entropia_condicional_dj += (len(Y_in_bin) / N) * bin_entropy

        # Calcular la ganancia de información para la variable 'j'
        ig_j = entropia_y - entropia_condicional_dj
        valores_ig.append((j, ig_j))

    # Ordenar las variables por ganancia de información en orden descendente
    valores_ig = sorted(valores_ig, key=lambda x: x[1], reverse=True)

    # Extraer los índices de las variables ordenadas y seleccionar las 'top_k' variables
    indices_ordenados = [idx + 1 for idx, _ in valores_ig]
    selected_features_indices = [idx + 1 for idx, _ in valores_ig[:top_k]]

    print(f"Top-{top_k} variables seleccionadas: {selected_features_indices}")
    for idx, ig in valores_ig:
        print(f"IG variable {idx + 1}: {ig}")

    # Crear una nueva base de datos reducida con las 'top_k' variables seleccionadas
    data_reduced = data[:, [idx - 1 for idx in selected_features_indices]]

    return indices_ordenados, data_reduced

# Load dataClass 
def load_data(file_path):   
    # Cargar los datos desde el archivo CSV
    data = np.loadtxt(file_path, delimiter=',', dtype=float)
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
    k_indices, caracteristicas_reduced = inform_gain(caracteristicas, clases, m, t, c, top_k)

    # Guardar todos los índices ordenados en Idx_variable.csv
    np.savetxt('Idx_variable.csv', k_indices, delimiter=',', fmt='%d')
    # Guardar la base de datos con solo las variables más relevantes en DataIG.csv
    np.savetxt('DataIG.csv', caracteristicas_reduced, delimiter=',', fmt='%f')

if __name__ == '__main__':   
	 main()

