import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from som_anomaly_detector.AnomalyDetection import AnomalyDetection # Our special Anomaly Detector
from mpl_toolkits.mplot3d import Axes3D

# Initialize our anomaly detector with some arbitrary paremeters.
# Note: in a production situation sk-learn's gridsearch could be used to determine optimal parameters.
anomaly_detector = AnomalyDetection((10, 10), 3, 8, 0.001, 2, 0.001, 10, 3) #Initialize the dector

# Lets generate some training and some evaluation data
# We take the training data from a mixture distribution
# The evaluation consists of 100 points taken from this mixture
# and 30 anomalies/noise

training = np.random.rand(1000,3)
training = np.vstack((training, np.random.rand(1000,3) + 10))

outliers = np.random.rand(15,3)*4-2
outliers_2 = np.random.rand(15,3)*4 + 8
evaluation = np.vstack((outliers, outliers_2, np.random.rand(50,3), np.random.rand(50,3) + 10))

# Fit the anomaly detector and apply to the evaluation data
anomaly_detector.fit(training,5000); # Fit the anomaly detector
anomaly_metrics = anomaly_detector.evaluate(evaluation) # Evaluate on the evaluation data

# We make a density plot and a histogram showing the distrbution
# of the number of points mapped to a BMU
plt.subplot(121)
density = gaussian_kde(anomaly_metrics)
xs = np.linspace(0,5,200)
plt.plot(xs,density(xs))

plt.subplot(122)
plt.hist(anomaly_detector.bmu_counts)
plt.show();

# We make two plots, one showing the real anomalies marked in red, one showing
# the points identified by the algorithm marked in red.
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(evaluation[0:29,0], evaluation[0:29,1], evaluation[0:29,2], c='red')
ax.scatter(evaluation[30:,0], evaluation[30:,1], evaluation[30:,2], c='black')

sec = fig.add_subplot(122, sharex=ax, sharey=ax, projection='3d')
selector = anomaly_metrics > 1
sec.scatter(evaluation[selector,0], evaluation[selector, 1], evaluation[selector, 2], c='red')
sec.scatter(evaluation[~selector, 0], evaluation[~selector, 1], evaluation[~selector, 2], c='black')
fig.show();
