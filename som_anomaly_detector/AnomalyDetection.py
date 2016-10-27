from KohonenSom import KohonenSom
from sklearn.neighbors import NearestNeighbors
import numpy as np


class AnomalyDetection(KohonenSom):
    """"
    This class uses provides an specific implementation of Kohonnen Som for anomaly detection.
    """
    def __init__(self, shape, input_size, learning_rate, learning_decay = 0.1, initial_radius = 1, radius_decay = 0.1,
                 minNumberPerBmu = 1, numberOfNeighbors = 3):
        super(AnomalyDetection, self).__init__(shape, input_size, learning_rate, learning_decay, initial_radius,
                                                radius_decay)

        self.minNumberPerBmu = minNumberPerBmu
        self.numberOfNeighbors = numberOfNeighbors
        return

    def get_bmu_counts(self, training_data):
        """
        This functions maps a training set to the fitted network and evaluates for each node in the SOM the number of
        evaluations mapped to that node. This gives counts per BMU.
        :param training_data: numpy array of training data
        :return: An array of the same shape as the network with the best matching units.
        """
        bmu_counts = np.zeros(shape=self.shape)
        for observation in training_data:
            bmu = self.get_bmu(observation)
            bmu_counts[bmu] += 1
        return bmu_counts

    def fit(self, training_data, num_iterations):
        """
        This function fits the anomaly detection model to some training data.
        It removes nodes that are too sparse by the minNumberPerBmu threshold.
        :param training_data: numpy array of training data
        :param num_iterations: number of iterations allowed for training
        :return: A vector of allowed nodes
        """
        self.reset()
        super(AnomalyDetection, self).fit(training_data, num_iterations)
        bmu_counts = self.get_bmu_counts(training_data)
        self.bmu_counts = bmu_counts.flatten()
        self.allowed_nodes = self.grid[bmu_counts >= self.minNumberPerBmu]
        return self.allowed_nodes

    def evaluate(self, evaluationData):
        """
        This function maps the evaluation data to the previously fitted network. It calculates the anomaly measure
        based on the distance between the observation and the K-NN nodes of this observation.
        :param evaluationData: Numpy array of the data to be evaluated
        :return: 1D-array with for each observation an anomaly measure
        """
        try:
            self.allowed_nodes
            assert self.allowed_nodes.shape[0] > 1
        except NameError:
            raise Exception("Make sure the method fit is called before evaluating data.")
        except AssertionError:
            raise Exception("There are no nodes satisfying the minimum criterium, algorithm cannot proceed.")
        else:
            classifier = NearestNeighbors(n_neighbors=self.numberOfNeighbors)
            classifier.fit(self.allowed_nodes)
            dist, _ = classifier.kneighbors(evaluationData)
        return dist.mean(axis=1)