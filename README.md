# SOM Anomaly Detection
This Python module provides a very simple, but efficient implementaion of Kohonen's Self-Organizing-Maps for anomaly
detection purposes. The idea is based on the following paper:

Tian, J., Azarian, M. H., Pecht, M. (2014). Anomaly detection using self-organizing maps-based K-nearest neighbour
Algorithm. _Proceedings of the European Conference of the Prognostics and Health Management Society._

## Simple description of the algorithm
1. Train a self organizing map of some dimension on the set of normal data (possibly containing some noise or outliers).
2. Per node in the SOM, count the number of training vectors mapped to this node, suppose we call this number the
_degree_.
3. Remove all nodes with _degree_ less than a certain threshold.
4. For each observation in the data to be assessed, perform k-NN wrt. the SOM nodes. And calculate the mean distance
to the nodes found. This is the _anomaly metric_.
5. Order the evaluation data wrt. to the _anomaly metric_.

# How to install
Installation can be done by executing:
	
	git clone https://github.com/FlorisHoogenboom/som-anomaly-detector.git

Subsequently one can install the package by just executing

	pip install

in the directory to which the repository was cloned.