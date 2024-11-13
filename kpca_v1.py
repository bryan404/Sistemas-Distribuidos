# Kernel-PCA using Gaussian function

import numpy as np

# Gaussian Kernel function
def kernel_gauss(X, sigma):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            distance = np.linalg.norm(X[i] - X[j]) ** 2
            K[i, j] = np.exp(-distance / (2 * sigma ** 2))
    return K

# Kernel-PCA
def kpca_gauss(X, sigma, top_k):
    # Step 1: Compute the Kernel Matrix
    K = kernel_gauss(X, sigma)
    
    # Step 2: Center the Kernel Matrix
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    
    # Step 3: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
    
    # Step 4: Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Step 5: Project the data onto the top-k principal components
    top_eigenvectors = eigenvectors[:, :top_k]
    X_kpca = K_centered @ top_eigenvectors
    
    return X_kpca

# Load data from DataIG.csv
def load_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    # Select the first 3000 samples
    data = data[:3000]
    # Assume the last column is the label (if applicable, adjust as necessary)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

# Main function
def main():
    # Load data
    file_path = 'DataIG.csv'
    X, y = load_data(file_path)
    
    # Parameters
    sigma = 6.5  # Adjust as needed
    top_k = 10  # Number of principal components to retain
    
    # Apply Kernel PCA
    X_kpca = kpca_gauss(X, sigma, top_k)
    
    # Save the new data to DataKpca.csv
    np.savetxt('DataKpca.csv', X_kpca, delimiter=',')
    print("KPCA completed and saved to DataKpca.csv")

if __name__ == '__main__':
    main()