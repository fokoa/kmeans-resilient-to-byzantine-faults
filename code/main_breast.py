#!usr/bin/python3
# -*- coding : utf8 -*-


import sys;
import getopt;
from mpi4py import MPI;

import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
from sklearn import decomposition;

from kmeans_resilient import KMeansResilient as KMR
from functions import save_indices_score, save_indices_score2

SEED = 7
AZIMUTH = -88
ELEVATION = 63
ROLL = 0


# MPI initialization
comm = MPI.COMM_WORLD;
P = comm.Get_size();
rank = comm.Get_rank();


def check_option(opt_arg):
    """ Check the arguments passed as parameters by the
        command prompt 

        Parameters :
        -----------
        opt_arg : str
            Arguments and options passed by the command prompt

        Return :
        -------
        opts : list
            Argument list 
        args : list
            Option list
    """

    try:
        opts, args = opt_arg;

    except getopt.GetoptError as err:
        print(err);
        print("Use :\t", sys.argv[0], "-b 5 \n\t",
            "or:", sys.argv[0], "--byzantine 5");
        sys.exit(-1);

    for opt, val in opts:

        if opt in ("-b", "--byzantine"):
            if val.isdigit() == False:
                raise ValueError("Enter an integer as number"
                                 "of byzantine machines.");

        elif opt in ("-h", "--help"):
            print("Use:", sys.argv[0], "-b 5\n",
            "or:", sys.argv[0], "--byzantine 5");
            sys.exit(-1);
        
        else:
            print("unhandled options");
            sys.exit(-1);

    return opts, args;


def check_Nbyzan(opts, P):
    """ Check and get the number of Byzantine machines that
        we are going to simulate 

        Parameters :
        -----------
        opts : str
            Options passed by the command prompt
        P : int
            Total number of machines (nodes or workers).
            1 coodinator ans the ramaining are workers

        Return :
        -------
        n_byzantines : int (entire natural)
            Number of byzantine machines that we
            are going to simulate
    """

    if len(opts) == 0:
        n_byzantines = 0;
        
    n_byzantines = int(opts[0][1]);
    
    if n_byzantines < 0 or n_byzantines > P - 1:
        raise ValueError("Number of byzantine must be an integer "
                         "< number of workers or >= 0");
           
    return n_byzantines;


def sort_centroides(centroids):
    """ Sort centroids according to their norms

        Parameters :
        -----------
        centroids : ndarray of shape (k, n_features)
            All centroids of clusters where k
            is number of clusters

        Return :
        -------
        tmp : ndarray of shape (k, n_features)
            Sorted centroids
    """

    tmp = np.zeros((centroids.shape));
    normes = {};

    for centroid in range(0, centroids.shape[0]):
        norm = np.linalg.norm(centroids[centroid]);
        normes[norm] = centroid;

    i=0;
    for norm in sorted(normes):
        tmp[i] = centroids[normes[norm]];
        i = i + 1;

    return tmp;
  

def comparaison_cluster(X, label_km, label_by, label_co, name_data, n_workers, n_byz):
    """ Plot all the formed clusters 

        Parameters :
        -----------
        X : ndarray of shape (n_samples, n_features)
            Samples to be clustered
        label_km : list of length 2
            The first is labels obtained with K-means
            The second is number of clusters
        label_by : list of length 2
            The first is labels obtained with byzantin K-means
            The second is number of byzantines
        label_co : ndarray of shape (n_samples, )
            Label obtained by correcting byzantines 
            in byzantin K-means
        name_data : ndarray of shape (n_samples, )
            Name of dataset we want to plot
        n_workers : int
            Number of workers
        n_byz : int
            Number of byzantines
    """

    pca = decomposition.PCA(n_components = 3);
    X_reduced = pca.fit_transform(X);

    x_axis = [val[0] for val in X_reduced];
    y_axis = [val[1] for val in X_reduced];
    z_axis = [val[2] for val in X_reduced];

    fig = plt.figure(figsize=plt.figaspect(0.5), facecolor="w");

    ax = fig.add_subplot(1, 3, 1, projection='3d');
    plt.title('%d-means'%(label_km[1]));
    ax.scatter(x_axis, y_axis, z_axis, c=label_km[0]);
    ax.view_init(ELEVATION, AZIMUTH, ROLL)

    ax = fig.add_subplot(1, 3, 2, projection='3d');
    plt.title('%d Byzantines' % (label_by[1]));
    ax.scatter(x_axis, y_axis, z_axis, c=label_by[0]);
    ax.view_init(ELEVATION, AZIMUTH, ROLL)

    ax = fig.add_subplot(1, 3, 3, projection='3d');
    ax.scatter(x_axis, y_axis, z_axis, c=label_co);
    ax.view_init(ELEVATION, AZIMUTH, ROLL)
    plt.title('Correction');

    plt.savefig(f"../figures/{name_data}_{n_workers}Workers_{n_byz}Byzantines.jpg", 
                format='jpg', dpi=500, bbox_inches='tight');
    # plt.show();


def comparaison_inertia(inertias_km, inertias_by, inertias_co, name_data, n_workers, n_byz):
    """ Plot all the formed clusters 

        Parameters :
        -----------
        inertias_km : list of length 2
            The first is inertia at each iteration obtained with K-means
            The second is number of clusters
        inertias_by : list of length 2
            The first is inertia at each iteration obtained with 
            byzantine K-means. The second is number of byzantines
        inertias_co : ndarray of shape (n_samples, )
            Inertia at each iteration obtained by correcting byzantines 
        name_data : ndarray of shape (n_samples, )
            Name of dataset we want to plot
        n_workers : int
            Number of workers
        n_byz : int
            Number of byzantines
    """
    x = np.arange(0, len(inertias_km[0]))
    plt.figure(figsize=(5,4), facecolor="w");
    plt.plot(x, inertias_km[0], label=f"{inertias_km[1]}-means (no byzantines)"); 
    plt.plot(x, inertias_by[0], label=f"{inertias_km[1]}-means ({inertias_by[1]} Byzantines)");
    plt.plot(x, inertias_co, label="correction");
    plt.xlabel("iteration")
    plt.ylabel("inertia")
    plt.legend()
    plt.title('%s & %d Byzantines'%(name_data, inertias_by[1]));

    plt.savefig(f"../figures/{name_data}_{n_workers}Workers_{n_byz}Byzantines_inertia.jpg", 
                format='jpg', dpi=500, bbox_inches='tight');
    # plt.show();


def main():

    # Check options and number of byzantines
    opts, arg = check_option(getopt.getopt(sys.argv[1:], "b:",
                             ["byzantine="]));
    n_byzantines = check_Nbyzan(opts, P);

    # Load dataset
    dataset_name = "breast"
    data = pd.read_csv("data/breast cancer/breast-cancer-wisconsin.data",
                       sep=',', header=None);
    data[6] = data[6].apply(lambda x : 0 if x == '?' else x);
    data[6] = data[6].astype(int);
    
    # Column to fit
    cols = data.columns.difference([0, 10]);
    X = data[cols].values;
    y = data[10].values;

    # Model
    km = KMR(n_clusters=2, n_init=1, n_iter=25, seed=SEED);
    by = KMR(n_clusters=2, n_init=1, n_iter=25, seed=SEED, n_byzantines=n_byzantines);
    co = KMR(n_clusters=2, n_init=1, n_iter=25, seed=SEED, n_byzantines=n_byzantines, correction=True);

    # Fit
    km.fit(X);
    by.fit(X, methode_gen='methode_1');
    co.fit(X, methode_gen='methode_1');

    # Sort centroides
    km.centroids_ = sort_centroides(km.centroids_);
    by.centroids_ = sort_centroides(by.centroids_);
    co.centroids_ = sort_centroides(co.centroids_);

    # Plot
    if rank == 0:

        # print('\nKmeans : \n' , km.centroids_);
        # print('Byzantine : \n', by.centroids_);
        # print('Correct : \n', co.centroids_);
        # print('\nKmeans inertia :\n', km.inertia_);
        # print('\nByzantine inertia :\n', by.inertia_);
        # print('\nCorrection inertia :\n', co.inertia_);

        comparaison_cluster(
                X,
                [km.labels_, km.n_clusters],
                [by.labels_, by.n_byzantines],
                co.labels_,
                "breast",
                P-1,
                n_byzantines
        )
        comparaison_inertia(
                [km.inertias_, km.n_clusters],
                [by.inertias_, by.n_byzantines],
                co.inertias_,
                "breast",
                P-1,
                n_byzantines
        )
        if len(np.unique(by.labels_))  > 1:
            save_indices_score(
                dataset_name,
                km.get_indices(X),
                by.get_indices(X),
                co.get_indices(X),
                n_byzantines
            )
        else:
            save_indices_score2(
                dataset_name,
                km.get_indices(X),
                co.get_indices(X),
                n_byzantines
            )


        """ if len(np.unique(by.labels_))  > 1:
            comparaison_indices(
                    [km.get_indices(X), km.n_clusters],
                    [by.get_indices(X), by.n_byzantines],
                    co.get_indices(X),
                    "breast",
                    P-1,
                    n_byzantines
            )
        else:
            comparaison_indices_2(
                    [km.get_indices(X), km.n_clusters],
                    co.get_indices(X),
                    "breast",
                    P-1,
                    n_byzantines
            ) """


if __name__ == "__main__":
    main();

