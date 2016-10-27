import numpy as np
import math
from scipy.stats import multivariate_normal

class KohonenSom(object):
    """
    This class provides an implementation of the Kohonen SOM algorithm. It supports SOMs of an arbitrary dimension,
    which may be handy for data quality purposes.
    """

    def __init__(self, shape, input_size, learning_rate, learning_decay = 1, initial_radius = 1, radius_decay = 1):
        """ Initialization of the SOM

        :param dimension: The shape of the network. Each entrty in the tuple corresponds to one direction
        :type dimension: tuple of ints
        :param learning_rate: The inital learning rate.
        :type learning_rate: float, should be > 0
        :param initial_radius:  The initial radius.
        :type initial_radius: float, should be > 0
        """

        self.shape = shape
        self.dimension = shape.__len__()
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.learning_decay = learning_decay
        self.initial_radius = initial_radius
        self.radius_decay = radius_decay

        # Initialize a distance matrix to avoid computing the distance on each iteration
        distance = np.fromfunction(self._distance_function, tuple(2*i + 1 for i in shape)) # We create some
        # redundant entries to make sure we have zero distance at index = shape, this is easier in later computations.

        gaussianTransorm = np.vectorize(lambda x: multivariate_normal.pdf(x, mean=0, cov=1))
        self.distance = gaussianTransorm(distance)

        # We add an extra dimension so that we can easily perform multiplication later on
        self.distance = np.repeat(self.distance, self.input_size, self.dimension-1)
        self.distance = np.reshape(self.distance, newshape=(distance.shape + (self.input_size,)))

        # Initialize the grid
        self.grid = np.random.rand(*(self.shape + (self.input_size,))) * 2 - 1

        return

    def reset(self):
        """
        This function resets the grid for a new estimation to take place
        :return: Nothing
        """
        self.grid = np.random.rand(*(self.shape + (self.input_size,))) * 2 - 1

        return


    def _distance_function(self, *args):
        """ Computes the euclidean distance for an arbitrary number of points
        :param points: arbitrary number of points
        :type points: float
        :return: the euclidean distance
        """

        return sum([(i-x)**2 for i,x in zip(args, self.shape)]) ## Fill the array in such a way it contains zero
        # distance at the center, i.e. index = shape

    def get_bmu(self, sample):
        """Find the best matchin unit for a specific sample

        :param sample: The data point for which the best matching unit should be found.
        :type sample: numpy.array
        :return: numpy.ndarray with index
        """

        distances = np.square(self.grid - sample).sum(axis=self.dimension)
        bmu_index = np.unravel_index(distances.argmin().astype(int), self.shape)
        return bmu_index

    def fit(self, training_sample, num_iterations):
        """Train the SOM to a specific dataset.

        :param training_sample: The complete training dataset
        :type training_sample: 2d ndarray
        :param num_iterations: The number of iterations used for training
        :type num_iterations: int
        :return: a reference to the object
        """

        sigma = self.initial_radius
        l = self.learning_rate
        for i in range(1, num_iterations):

            obs = training_sample[np.random.choice(training_sample.shape[0], 1)][0]
            bmu = self.get_bmu(obs)
            self.update_weights(obs, bmu, sigma, l)

            # Update the parameters to let them decay to 0
            sigma = self.initial_radius * math.exp(-(i*self.radius_decay))
            l = self.learning_rate * math.exp(-(i*self.learning_decay))
        return self

    def update_weights(self, training_vector, bmu, sigma, learning_speed):
        reshaped_array = self.grid.reshape((np.product(self.shape), self.input_size))

        # We want to roll the distance matrix such that we have the BMU at the center
        bmuDistance = self.distance
        for i,bmu_ind in enumerate(bmu):
            bmuDistance = np.roll(bmuDistance, bmu_ind, axis=i)

        # Next we take part of the second quadrant of the matrix since this corresponds to the distance matrix we desire
        for i, shape_ind in enumerate(self.shape):
            slc = [slice(None)]*len(bmuDistance.shape)
            slc[i] = slice(shape_ind, 2*shape_ind)
            bmuDistance = bmuDistance[slc]

        # Multiply by sigma to emulate a decreasing radius effect
        bmuDistance = sigma*bmuDistance


        learningMatrix = -(self.grid - training_vector)
        scaledLearningMatrix = learning_speed * (bmuDistance * learningMatrix)
        self.grid = self.grid + scaledLearningMatrix

        return