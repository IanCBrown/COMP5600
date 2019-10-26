import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

def distance(p, q):
    return np.linalg.norm(p - q, axis=1)
    

def get_data(file_name):
    data = []
    for row in open(file_name):
        r = row.split()
        data.append((float(r[0]), float(r[1])))
    return data

def main():
    data = get_data("B.txt")

    cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='single')  
    cluster.fit_predict(data)
    
    # plt.scatter(*zip(*data), color='blue')
    # plt.figure(figsize=(10, 7))  
    # plt.title("Dendrograms")  
    # plt.scatter(*zip(*data), c=cluster.labels_, cmap='rainbow')
    dend = shc.dendrogram(shc.linkage(data, method='single'))
    # plt.scatter(*zip(*centroids), color='black')
    plt.show()



if __name__ == "__main__":
    main()
    