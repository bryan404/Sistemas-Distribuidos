#-------------------------------------------------------
# Load Data from File: KDDTrain.txt
#--------------------------------------------------------

import numpy as np
#import utility_etl  as ut                                  # <---------------- ¿Que es esto?

# Load parameters from config.csv                           # <---------------- ¿Que es esto?
def config():
    param = np.loadtxt("config.csv", delimiter=",", dtype = str)
    print(f'Information: {param}')

# Beginning ...
def main():
    # Inicializamos los datos necesarios. La información la obtivimos desde el siguiente link: https://www.kaggle.com/datasets/hassan06/nslkdd?select=KDDTrain%2B.arff
    protocol_type = ['tcp','udp', 'icmp']
    service = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50']
    flag = [ 'OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH' ]
    attackDOS = ['neptune','teardrop','smurf','pod','back','land','apache2','processtable', 'mailbomb','udpstorm']
    attackProbe = ['ipsweep','portsweep','nmap','satan','saint','mscan']

    # Guardamos los datos en una variable "data"
    data = np.loadtxt("KDDTrain.txt", delimiter=",", dtype = str)
    print("------------------DATA---------------------")
    print(f'Size: {data.size}')
    print(f'dimension: {data.shape}')

    # Copiamos la "data" pero le quitamos la última columna
    auxData = np.array(data[:, :-1], copy=True)

    # Iniciamos las matrices
    class1 = []
    class2 = []
    class3 = []

    # -> Ahora transformamos los datos originales en formato numérico <-
    for i in range(0,auxData.shape[0]):
        # Tranformamos el dato "protocol_type" a numérico
        for k in range(len(protocol_type)):
            if auxData[i][1] == protocol_type[k]:
                auxData[i][1] = k
        # Tranformamos el dato "service" a numérico
        for k in range(len(service)):
            if auxData[i][2] == service[k]:
                auxData[i][2] = k
        # Tranformamos el dato "flag" a numérico
        for k in range(len(flag)):
            if auxData[i][3] == flag[k]:
                auxData[i][3] = k
        
        # Tranformamos el dato "class" que corresponde al ataque a numérico
        # Si el ataque es "normal" se guarda como "1"
        if auxData[i][41] == 'normal':
            auxData[i][41] = 1
            class1.append(auxData[i][:-1])
            pass
        # Si el ataque es "Probe" se guarda como "3"
        for k in range(len(attackProbe)):
            if auxData[i][41] == attackProbe[k]:
                auxData[i][41] = 3
                class3.append(auxData[i][:-1])
                pass
        # Si el ataque es "DOS" se guarda como "2"
        """ auxData[i][41] = 2
        class2.append(auxData[i]) """
        for k in range(len(attackDOS)):
            if auxData[i][41] == attackDOS[k]:
                auxData[i][41] = 2
                class2.append(auxData[i][:-1])
            
    # -> Ver datos <- (Se puede eliminar)
    """ print("------------------auxData---------------------")
    print(f'Size: {auxData.size}')
    print(f'dimension: {auxData.shape}')
    print(f'Information modify: {auxData}')

    print("------------------class1---------------------")
    print(f'dimension: {len(class1)}')

    print("------------------class2---------------------")
    print(f'dimension: {len(class2)}')

    print("------------------class3---------------------")
    print(f'dimension: {len(class3)}') """
    
    # Guardamos todos los datos en sus archivos correspondientes.
    np.savetxt("Data.csv", auxData, delimiter=",", fmt='%s')
    np.savetxt("class1.csv", class1, delimiter=",", fmt='%s')
    np.savetxt("class2.csv", class2, delimiter=",", fmt='%s')
    np.savetxt("class3.csv", class3, delimiter=",", fmt='%s')

    # -> Ahora procederemos a crear el archivo de clases con las muestras. <-

    # Obtenemos los índices de los archivos idx_class
    idx_class1 = np.loadtxt("idx_class1.csv", delimiter=",", dtype = int)
    idx_class2 = np.loadtxt("idx_class2.csv", delimiter=",", dtype = int)
    idx_class3 = np.loadtxt("idx_class3.csv", delimiter=",", dtype = int)

    # Creamos matriz con datos de muestras
    dataM = []

    # Agregamos datos a dataM
    for i in range(len(idx_class1)):
        dataM.append(auxData[idx_class1[i]-1])
    for i in range(len(idx_class2)):
        dataM.append(auxData[idx_class2[i]-1])
    for i in range(len(idx_class3)):
        dataM.append(auxData[idx_class3[i]-1])
    
    # Guardamos los datos en un archivo csv
    np.savetxt("DataClass.csv", dataM, delimiter=",", fmt='%s')


    #config()            
   
      
if __name__ == '__main__':   
	 main()

