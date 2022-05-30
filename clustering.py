import numpy as np

from sklearn.cluster import DBSCAN, OPTICS,MeanShift
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from common_utils import raster_proc as rproc




band_files = ["D:\\work\\demo_projects\\soil_clusters\\S2B_L2A_2021-06-18_121040_34UEF_mosaic\\S2B_L2A_2021-06-18_121040_34UEF_B02_mosaic.tif",
                "D:\\work\\demo_projects\\soil_clusters\\S2B_L2A_2021-06-18_121040_34UEF_mosaic\\S2B_L2A_2021-06-18_121040_34UEF_B03_mosaic.tif",
                "D:\\work\\demo_projects\\soil_clusters\\S2B_L2A_2021-06-18_121040_34UEF_mosaic\\S2B_L2A_2021-06-18_121040_34UEF_B04_mosaic.tif"]
fields_file = "D:\\work\\demo_projects\\soil_clusters\\border (6).geojson"

output_file = "D:\\work\\demo_projects\\soil_clusters\\clusters_ms_0075.tif"


srs,geotr = rproc.extract_georeference(band_files[0],cutline=fields_file)
img = rproc.open_clipped_raster_as_image(band_files[0],cutline=fields_file)
pix_data = np.full(((img.shape[0]*img.shape[1]),3),fill_value=0,dtype=float)

i = 0
for b in band_files:
    img = rproc.open_clipped_raster_as_image(b, cutline=fields_file, dst_nodata=0)
    pix_data[:,i] = np.reshape(img,img.shape[0]*img.shape[1])/10000
    i+=1

pix_data_mask = (pix_data[:,0]!=0) & (pix_data[:,1]!=0) & (pix_data[:,2]!=0)
input_clustering = pix_data[pix_data_mask]

#clastered_data = DBSCAN(eps=0.3, min_samples=1000).fit(input_clustering)
#clustered_data = OPTICS(min_samples=100, xi=0.3, min_cluster_size=0.3).fit(input_clustering)
clustered_data =  MeanShift(bandwidth=0.0075, bin_seeding=True).fit(input_clustering)

cluster_labels = clustered_data.labels_ + 1


output_img = np.full((img.shape[0]*img.shape[1]),fill_value=0,dtype=np.uint16)
output_img[pix_data_mask] = cluster_labels
output_img = np.reshape(output_img,(img.shape[0],img.shape[1]))

#for i in range(len(band_files)):
#    output_img[i,:,:] = np.reshape(data[:,i],(img.shape[0],img.shape[1]))
rproc.array2geotiff(output_file,[geotr[0],geotr[3]],geotr[1],srs,output_img)

print('OK')


#srs,geotr = rproc.extract_georeference(B4_file,cutline=fields_file)
#rproc.array2geotiff("D:\\work\\demo_projects\\soil_clusters\\1.tif",[geotr[0],geotr[3]],geotr[1],srs,img,nodata_val=0)
#exit(0)



# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)

X = StandardScaler().fit_transform(X)

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print(
    "Adjusted Mutual Information: %0.3f"
    % metrics.adjusted_mutual_info_score(labels_true, labels)
)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()