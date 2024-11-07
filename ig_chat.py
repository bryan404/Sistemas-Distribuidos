import numpy as np

def create_embedding_vectors(X, m, tau, M):
    # Crear una lista de los vectores embedding
    embedding_vectors = [X[i:i + m * tau:tau] for i in range(M)]
    return np.array(embedding_vectors)

# Función para normalizar el vector de datos X entre 0 y 1
def normalizar_datos(x):
    mu_x = np.mean(x)  # Media de la variable x
    sigma_x = np.std(x)  # Desviación estándar de la variable x
    epsilon = 1e-10  # Pequeña constante para evitar divisiones por cero
    
    # Paso 1: Calcular u
    u = (x - mu_x) / (sigma_x + epsilon)
    
    # Paso 2: Calcular X normalizado
    X_normalizado = 1 / (1 + np.exp(-u))
    
    return X_normalizado

def convertir_a_patron(Y, base_c, m):
    vector_c = np.array([base_c**i for i in range(m)])    
    # Cálculo del valor de k
    k = 1 + np.dot(Y - 1, vector_c)
    return k

def calcular_Y(X, c):
    # Aplica la fórmula Yi = round(c * Xi + 0.5) a cada elemento de X
    Y = np.round(c * X + 0.5)
    return Y

def calcular_entropia_dispersion(p, base_c, m):
    # Calcular r = c^m
    r = base_c ** m
    
    # Asegurar que el vector de probabilidades suma 1 y no contiene ceros
    p = np.array(p)
    p = p / np.sum(p)
    p = p + (p == 0) * 1e-10  # Añadir una pequeña constante para evitar log(0)
    
    # Calcular la Entropía de Dispersión
    DE = -np.sum(p * np.log2(p))
    
    return DE

# Ejemplo de uso
m = 3  # Dimensión de embeding
tau = 2  # Tiempo de retraso
c = 3

x = np.random.randint(1, 3, (10, 3))  # Vector de datos de ejemplo

N = len(x)
M = N - (m - 1) * tau 

X_normalizado = normalizar_datos(x)

print("Datos normalizados:", X_normalizado)

# Crear los vectores embedding
embedding_vectors = create_embedding_vectors(X_normalizado, m, tau, M)
print("Vectores embedding:", embedding_vectors)

Y = calcular_Y(embedding_vectors, c)
print("Y_i: ", Y)

patron = convertir_a_patron(Y, c, m)
print("Patrón: ", patron)

# Calcular la frecuencia de ocurrencia de cada patrón k
unique_k, counts_k = np.unique(patron, return_counts=True)

print("Frecuencia de ocurrencia de los patrones k:", counts_k)

# Calcular la probabilidad de ocurrencia de cada patrón k
p = counts_k / counts_k.sum()
print("Probabilidad de ocurrencia de los patrones K",p)

# Calcular la entropía de dispersión usando las probabilidades calculadas
entropy = calcular_entropia_dispersion(p, c, m)

print("Entropía de los patrones de dispersión:", entropy)

print("Entropía normalizada:", entropy / np.log2(c ** m))