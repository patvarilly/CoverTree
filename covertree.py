# Copyright Patrick Varilly 2012
# Released under the scipy license
#
# Based on PyCoverTree (http://github.com/emanuele/PyCoverTree), as
# modified by Emanuele Olivetti, license as follows:
# 
# File: covertree.py
# Date of creation: 05/04/07
# Copyright (c) 2007, Thomas Kollar <tkollar@csail.mit.edu>
# Copyright (c) 2011, Nil Geisweiller <ngeiswei@gmail.com>
# All rights reserved.
#
# This is a class for the cover tree nearest neighbor algorithm.  For
# more information please refer to the technical report entitled "Fast
# Nearest Neighbors" by Thomas Kollar or to "Cover Trees for Nearest
# Neighbor" by John Langford, Sham Kakade and Alina Beygelzimer
#  
# If you use this code in your research, kindly refer to the technical
# report.

# I have rewritten the code to mimic the API of Anne M. Archibald's
# scipy.spatial.kdtree

import numpy as np

__all__ = ['CoverTree', 'distance_matrix']

class CoverTree(object):
    """
    Cover tree for quick nearest-neighbor lookup in general metric spaces.

    TODO: Write more
    """
    def __init__(self, data, distance):
        """
        Construct a cover tree.

        Parameters
        ----------
        data : array_like, shape (n,) + pt_shape
            The data points to be indexed. This array is not copied, so
            modifying this data will result in bogus results. Point i is
            stored in data[i].  If pt_shape is not (,), then data[i] is a
            smaller numpy array, which is useful when defining points using
            coordinates.
        distance : two-argument callable returning a float
            Given two points p and q, return the distance d between them.
            d(p,q) must be a metric, meaning that
            * d(p,q) >= 0
            * d(p,q) = 0  iff p == q
            * d(p,q) = d(q,p)
            * q(p,q) <= d(p,r) + d(r,q) for all r
            'Points' here means elements of the data array.

        Examples
        --------

        Two 3D points in a CoverTree using squared Euclidean distance
        as a metric
        
        >>> data = np.array([[0,0,0], [1.5,2.3,4.7]])
        >>> ct = CoverTree(data, scipy.spatial.distance.euclidean)

        Two strings in a CoverTree using Levenshtein (edit) distance.
        [The implementation here, from Wikibooks, has terrible performance,
        but illustrates the idea cleanly]

        >>> def lev(a, b):
                if not a: return len(b)
                if not b: return len(a)
                return min(lev(a[1:], b[1:])+(a[0] != b[0]),
                           lev(a[1:], b)+1, lev(a, b[1:])+1)
        >>> data = np.array(['hello', 'halo'])
        >>> ct = CoverTree(data, lev)
        """
        raise NotImplementedError

    def query(self, x, k=1, eps=0, distance_upper_bound=np.inf):
        """
        Query the cover tree for nearest neighbors

        Parameters
        ----------
        x : array_like, shape tuple + pt_shape
            An array of points to query.
        k : integer
            The number of nearest neighbors to return.
        eps : nonnegative float
            Return approximate nearest neighbors; the kth returned value
            is guaranteed to be no further than (1+eps) times the
            distance to the real kth nearest neighbor.
        distance_upper_bound : nonnegative float
            Return only neighbors within this distance. This is used to
            prune tree searches, so if you are doing a series of
            nearest-neighbor queries, it may help to supply the distance to
            the nearest neighbor of the most recent point.

        Returns
        -------
        d : array of floats
            The distances to the nearest neighbors.
            If x has shape tuple + pt_shape, then d has shape tuple if
            k is one, or tuple+(k,) if k is larger than one.  Missing
            neighbors are indicated with infinite distances.  If k is None,
            then d is an object array of shape tuple, containing lists
            of distances. In either case the hits are sorted by distance
            (nearest first).
        i : array of integers
            The locations of the neighbors in self.data. i is the same
            shape as d.
        """
        raise NotImplementedError
    
    def query_ball_point(self, x, r, eps=0):
        """Find all points within distance r of point(s) x.

        Parameters
        ----------
        x : array_like, shape tuple + pt_shape
            The point or points to search for neighbors of.
        r : positive float
            The radius of points to return.
        eps : nonnegative float, optional
            Approximate search. Branches of the tree are not explored if
            their nearest points are further than ``r / (1 + eps)``, and
            branches are added in bulk if their furthest points are nearer
            than ``r * (1 + eps)``.

        Returns
        -------
        results : list or array of lists
            If `x` is a single point, returns a list of the indices of the
            neighbors of `x`. If `x` is an array of points, returns an
            object array of the same shape as `x` containing lists of
            neighbors.

        Notes
        -----
        If you have many points whose neighbors you want to find, you may
        save substantial amounts of time by putting them in a CoverTree and
        using query_ball_tree.
        
        """
        raise NotImplementedError
    
    def query_ball_tree(self, other, r, eps=0):
        """
        Find all pairs of points whose distance is at most r

        Parameters
        ----------
        other : CoverTree
            The tree containing points to search against.  Its distance
            function must be identical to self.distance.
        r : positive float
            The maximum distance
        eps : nonnegative float
            Approximate search. Branches of the tree are not explored if
            their nearest points are further than r/(1+eps), and branches
            are added in bulk if their furthest points are nearer than
            r*(1+eps).

        Returns
        -------
        results : list of lists
            For each element self.data[i] of this tree, results[i] is a list
            of the indices of its neighbors in other.data.
        """
        raise NotImplementedError

    def query_pairs(self, r, eps=0):
        """
        Find all pairs of points whose distance is at most r

        Parameters
        ----------
        r : positive float
            The maximum distance
        eps : nonnegative float
            Approximate search. Branches of the tree are not explored if
            their nearest points are further than r/(1+eps), and branches
            are added in bulk if their furthest points are nearer than
            r*(1+eps).

        Returns
        -------
        results : set
            set of pairs (i,j), i<j, for which the corresponing positions
            are close.
        """
        raise NotImplementedError
    
    def count_neighbors(self, other, r):
        """
        Count how many nearby pairs can be formed.

        Count the number of pairs (x1,x2) that can be formed, with x1 drawn
        from self and x2 drawn from other, and where d(x1,x2) <= r.
        This is the "two-point correlation" described in Gray and Moore
        2000, "N-body problems in statistical learning", and the code here
        is based on their algorithm.

        Parameters
        ----------
        other : CoverTree
            The tree containing points to search against.  Its distance
            function must be identical to self.distance.
        r : float or one-dimensional array of floats
            The radius to produce a count for. Multiple radii are searched
            with a single tree traversal.

        Returns
        -------
        result : integer or one-dimensional array of integers
            The number of pairs. Note that this is internally stored in a
            numpy int, and so may overflow if very large (two billion).
        """
        raise NotImplementedError

    def sparse_distance_matrix(self, other, max_distance):
        """
        Compute a sparse distance matrix

        Computes a distance matrix between two CoverTrees, leaving as zero
        any distance greater than max_distance.

        Parameters
        ----------
        other : CoverTree
            The tree containing points to search against.  Its distance
            function must be identical to self.distance.
        max_distance : positive float

        Returns
        -------
        result : dok_matrix
            Sparse matrix representing the results in "dictionary of keys"
            format.
        """
        raise NotImplementedError

def distance_matrix(x,y,distance):
    """
    Compute the distance matrix.

    Returns the matrix of all pair-wise distances.

    Parameters
    ----------
    x : array_like, shape (M,) + pt_shape
        The first set of m points
    y : array_like, shape (N,) + pt_shape
        The second set of n points
    distance : two-argument callable returning float
        distance(p,q) returns the distance between points p and q

    Returns
    -------
    result : array_like, `M` by `N`
    """

    x = np.asarray(x)
    m = x.shape[0]
    pt_shape_x = x.shape[1:]
    y = np.asarray(y)
    n = y.shape[0]
    pt_shape_y = y.shape[1:]

    if pt_shape_x != pt_shape_y:
        raise ValueError("x contains vectors of shape %s but y contains "
                         "vectors of shape %s"
                         % (str(pt_shape_x), str(pt_shape_y)))

    result = np.empty((m,n),dtype=np.float)
    for i,j in np.ndindex((m,n)):
        result[i,j] = distance(x[i],y[j])
    return result
