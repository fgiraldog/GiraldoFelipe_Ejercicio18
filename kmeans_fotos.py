  
from matplotlib import image
import matplotlib.pyplot as plt
import glob
import numpy as np
import sklearn.cluster

imagenes = []
names = glob.glob("imagenes/*.png")



for element in names:
    image_ = np.float_(image.imread(element))
    imagenes.append(image_)

imagenes = np.array(imagenes).reshape(len(imagenes),-1)
clusters = np.arange(1,20,1)
inertia = []

for i in clusters:
	k_means = sklearn.cluster.KMeans(n_clusters=i)
	k_means.fit(imagenes)
	inertia.append(k_means.inertia_)

plt.figure()
plt.plot(clusters, inertia)
plt.xlabel('# of clusters')
plt.ylabel('Inertia')
plt.xticks(clusters)
plt.tight_layout()
plt.savefig('inercia.png')


#se ve un claro cambio de pendiente en 5 clusters
best_c = 5
best_k = sklearn.cluster.KMeans(n_clusters=best_c)
best_k.fit(imagenes)
centers = best_k.cluster_centers_

#Estos fors no se demoran (lo prometo), el que se demora es el anterior

plt.figure(figsize = (8,8))
for j in range(0,len(centers)):
	distances = []
	for i in range(0,len(names)):
		distances.append(np.sqrt(np.sum(np.square(imagenes[i,:]-centers[j,:]))))

	sorted_index = np.argsort(distances)
	best_index = sorted_index[:5]

	for k in range(0,len(best_index)):
		plt.subplot(5,5,(j*5)+(k+1))
		plt.imshow(imagenes[best_index[k],:].reshape((100,100,3)))
	plt.title('K = {}'.format(j+1))

plt.tight_layout()
plt.savefig('ejemplo_clases.png')

