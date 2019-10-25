import numpy as np
import matplotlib.pyplot as plt
import math


def distance(p, q):
    return np.linalg.norm(p - q, axis=1)
    

def get_data(file_name):
    data = []
    for row in open(file_name):
        r = row.split()
        data.append((float(r[0]), float(r[1])))
    return data

def main():
    data = get_data("A.txt")
    k = 3
    # Use random centroids
    # X
    C_x = [np.random.uniform(0, np.max(data)) for i in range(k)]
    # Y 
    C_y = [np.random.uniform(0, np.max(data)) for i in range(k)]
    centroids = np.array(list(zip(C_x, C_y)))
    print(centroids)

    C_prev = np.zeros(centroids.shape)
    # k cluster labels
    clusters = np.zeros(len(data))
    # error 
    error = distance(centroids, C_prev)

    while error.all() != 0:
        for i in range(len(data)):
            distances = distance(data[i], centroids)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        
        C_prev = centroids
        # recalculate centroids
        # 0, 1, 2...
        for cluster in range(k):
            points_in_class = [data[i] for i in range(len(data)) if clusters[i] == cluster]
            # centroids[int(classification)] = np.average([x for x in clusters if x == classification])
            centroids[cluster] = np.mean(points_in_class, axis=0)

        error = distance(centroids, C_prev)


    markers = ["d", "s", "o"]
    colors = ['red','blue','green']
    for cluster in range(k):
        points_in_class = [data[i] for i in range(len(data)) if clusters[i] == cluster]
        plt.scatter(*zip(*points_in_class), color=colors[cluster], marker=markers[cluster])

    plt.scatter(*zip(*centroids), color='black')
    plt.show()



if __name__ == "__main__":
    main()
    