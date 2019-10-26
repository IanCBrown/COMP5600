import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import combinations
from queue import PriorityQueue
from functools import reduce

def distance(p, q):   
    return np.sqrt(np.sum((np.asarray(p)-np.asarray(q))**2))

def get_data(file_name):
    data = []
    for row in open(file_name):
        r = row.split()
        data.append((float(r[0]), float(r[1])))
    return data


def get_closest_points(distance_queue):
    # return np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)
    return distance_queue.get()

def merge(clusters, centroids, closest_points, data, label):
    # new = new centroid
    new = ((closest_points[1][0] + closest_points[2][0])/2, (closest_points[2][1] + closest_points[2][1])/2)
    for point, i in enumerate(data):
        if np.allclose(closest_points[1], point) or np.allclose(closest_points[2], point):
            clusters[i] = label
            centroids[i] = new 

def generate_distance_queue(data):
    # clusters = list(cluster_map.keys())
    combos = combinations(data, 2)
    ret = PriorityQueue()
    for c in combos:
        ret.put((distance(c[0], c[1]), c[0],c[1]))
    return ret


def main():
    data = get_data("B.txt")
    plt.scatter(*zip(*data), color='blue')
    print("Running...")

    # label
    iteration = 0 
    cluster_count = len(data) # or 2
    clusters = [i for i in range(cluster_count)]
    centroids = []
    cluster_map = {data[i] : iteration for i in range(len(data))}

    distance_queue = generate_distance_queue(data)


    # n_clusters = 2
    # affinity = euclidean 
    # linkage = single 
    while cluster_count > 1:
        # merge closest two clusters
        closest_points = distance_queue.get()

        new = ((closest_points[1][0] + closest_points[2][0])/2, (closest_points[2][1] + closest_points[2][1])/2)

        # merge the clusters 
        i = 0
        for point in data:
            if np.allclose(closest_points[1], point) or np.allclose(closest_points[2], point):
                clusters[i] = iteration
                centroids.append(new)
            i += 1

        # distance_queue = generate_distance_queue(centroids)

        cluster_map[closest_points[1]] = iteration
        cluster_map[closest_points[2]] = iteration
            
        cluster_count -= 1
        iteration += 1
    
    # uncomment to attempt to plot 
    # colors = ['red', 'blue', 'green']
    # for i in range(len(clusters)):
    #     plt.scatter(data[i][0], data[i][1], color=colors[clusters[i] - 1])


    print(clusters)
    plt.scatter(*zip(*centroids), color='black')
    plt.show()



if __name__ == "__main__":
    main()
    