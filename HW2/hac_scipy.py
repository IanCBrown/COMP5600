import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.cluster.hierarchy as shc
import scipy.spatial.distance as ssd 
from itertools import combinations

def distance(p, q):
    return np.sqrt(np.sum((np.asarray(p)-np.asarray(q))**2))
    

def get_data(file_name):
    data = []
    for row in open(file_name):
        r = row.split()
        data.append((float(r[0]), float(r[1])))
    return data

def generate_distance_matrix(data):
    # clusters = list(cluster_map.keys())
    n = len(data)
    ret = np.zeros((n,n), float)
    for i in range(n):
        for j in range(n):
            ret[i][j] = distance(data[i], data[j])
    return ret

def main():
    data = get_data("A.txt")
    
    d = generate_distance_matrix(data)
    distance_array =ssd.squareform(d) 
    linkage_matrix = shc.linkage(distance_array, method='single')


    clusters = shc.fcluster(linkage_matrix, 3, 'maxclust')
    colors = ['red', 'blue', 'green']
    marks = ['d', 's', 'o']
    for i in range(len(clusters)):
        plt.scatter(data[i][0], data[i][1], color=colors[clusters[i] - 1], marker=marks[clusters[i] - 1])
 
    # dend = shc.dendrogram(shc.linkage(data, method='single'))
    # plt.scatter(*zip(*centroids), color='black')
    plt.show()



if __name__ == "__main__":
    main()
    