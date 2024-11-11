import numpy as np

# Paso 1: Calcular la Matriz del Kernel
def calcular_kernel(X, sigma):
    """
    Calcula la matriz del kernel usando el kernel gaussiano.
    """
    # Calcula la distancia euclidiana al cuadrado entre cada par de puntos
    dist_sq = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
    K = np.exp(-dist_sq / (2 * sigma ** 2))
    return K

# Paso 2: Centrar la Matriz del Kernel
def centrar_kernel(K):
    """
    Centra la matriz del kernel para que tenga media cero.
    """
    N = K.shape[0]
    uno_N = np.ones((N, N)) / N
    K_centrado = K - uno_N @ K - K @ uno_N + uno_N @ K @ uno_N
    return K_centrado

# Paso 3: Calcular los Valores y Vectores Propios
def calcular_valores_vectores_propios(K):
    """
    Calcula los valores y vectores propios de la matriz del kernel centrado.
    """
    valores_propios, vectores_propios = np.linalg.eigh(K)
    # Ordenar en orden descendente
    indices_ordenados = np.argsort(valores_propios)[::-1]
    valores_propios = valores_propios[indices_ordenados]
    vectores_propios = vectores_propios[:, indices_ordenados]
    return valores_propios, vectores_propios

# Paso 4: Seleccionar los k Componentes Principales
def proyectar_datos(K_centrado, vectores_propios, k):
    """
    Proyecta los datos sobre los k componentes principales.
    """
    U_k = vectores_propios[:, :k]
    return K_centrado @ U_k

# Ejemplo de uso
if __name__ == "__main__":
    # Datos de ejemplo
    X = np.loadtxt('DataIG.csv', delimiter=',', dtype=float)[:3000]

    # Parámetro de ancho del kernel
    sigma = 6.5
    
    # 1. Calcular la matriz del kernel
    K = calcular_kernel(X, sigma)

    # 2. Centrar la matriz del kernel
    K_centrado = centrar_kernel(K)
    
    # 3. Calcular valores y vectores propios
    valores_propios, vectores_propios = calcular_valores_vectores_propios(K_centrado)
    
    # 4. Proyectar sobre los k componentes principales (por ejemplo, k=2)
    k = 10
    X_proyectado = proyectar_datos(K_centrado, vectores_propios, k)
    
    print("Matriz del Kernel centrado:")
    print(K_centrado)
    print("\nValores propios:")
    print(valores_propios)
    print("\nVectores propios:")
    print(vectores_propios)
    print("\nDatos proyectados sobre los componentes principales:")
    print(X_proyectado)
