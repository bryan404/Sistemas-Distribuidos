# Information Gain

import numpy as np

# Dispersion entropy
def entropy_disp():   
    #....
    return()

# Normalised by use sigmoidal
def norm_data_sigmoidal(data):
    # Calcular la media y la desviación estándar de los datos
    mean = np.mean(data)
    std = np.std(data)
    
    # Aplicar la normalización sigmoidal con el exponente ajustado
    return 1 / (1 + np.exp(-(data - mean) / (std + 1e-10)))

# Information gain
def inform_gain():    
    #....
    return()

# Load dataClass 
def load_data(file_path):
    # Cargar los datos desde el archivo CSV
    data = np.loadtxt(file_path, delimiter=",", dtype=float)
    return data

# Beginning ...
def main():
    # Rutas a los archivos
    data_file_path = "./DataClass.csv"
    config_file_path = "./config.csv"
    
    # Cargar los datos
    data = load_data(data_file_path)
    
    # Normalizar los datos
    normalized_data = norm_data_sigmoidal(data)
    
    # Cargar la configuración
    m, t, c = np.loadtxt(config_file_path, delimiter=",", dtype=int, max_rows=3)
    
    # Crear los vectores-embedding
    N = len(normalized_data)
    M = N - (m - 1) * t
    embedding_vectors = np.array([normalized_data[i:i + m * t:t] for i in range(M)])    

    symbol_vectors = np.round(c * embedding_vectors + 0.5).astype(int)
    print("Muestra de vectores-embedding:")
    print(embedding_vectors)

    print("Muestra de vectores de símbolos:")
    print(symbol_vectors)
if __name__ == '__main__':   
    main()

