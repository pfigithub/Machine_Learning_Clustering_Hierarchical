# importing packages
import numpy as np 
import pandas as pd 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn.cluster import AgglomerativeClustering 

# reading
pdf = pd.read_csv('yourdata')
pdf.head(5)