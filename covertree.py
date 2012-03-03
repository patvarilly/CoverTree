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
from collections import defaultdict
import operator
import math
import itertools
import sys
import heapq

__all__ = ['CoverTree', 'distance_matrix']

class CoverTree(object):
    """
    Cover tree for quick nearest-neighbor lookup in general metric spaces.

    TODO: Write more
    """

    # Sentinel values for the C_infty and C_-infty levels
    _HIGHEST_LEVEL = 2**30-1
    _LOWEST_LEVEL = -2**30

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
        self.data = np.asarray(data)
        self.n = self.data.shape[0]
        self.pt_shape = self.data.shape[1:]
        self.distance = distance
        self.tree = self._build()

    class _Node(object):
        """
        A node in the cover tree.

        In the implicit representation, each node in the tree has a
        fixed level i, an associated point p and a list of children
        in level i-1.  If a point p first appears at level i, then
        a node corresponding to p appears at every lower level.

        In the explicit representation used here, each point p is associated
        with a single node at the highest level i.  Instead of having a
        single list of children, each node keeps lists of children of the
        implicit nodes associated with p at all levels j < i.
        """
        def __init__(self, idx, highest_level):
            """Create an explicit node for the point p = tree.data[idx] that
            first appears at level highest_level."""
            self.idx = idx
            self.highest_level = highest_level
            # dict mapping level and children
            self.children = defaultdict(list)
            self.highest_parent = None

        def __str__(self):
            return ('<id=%d, idx=%d, highest_level=%d, highest_parent=%d, children=...>'
                    % (id(self), self.idx, self.highest_level,
                       id(self.highest_parent)))

        def __repr__(self):
            return str(self)

        def add_child(self, child, level):
            """
            Add a child to the node associated with p at level ``level``.
            """
            assert level <= self.highest_level
            assert child.highest_parent == None
            self.children[level].append(child)
            child.highest_parent = self

        def get_children(self, level):
            """
            Get the children of the node associated with p at level
            ``level``.
            """
            assert level <= self.highest_level
            retLst = [self]
            if level in self.children:
                retLst.extend(self.children[level])
            return retLst

        def get_only_children(self, level):
            """Like get_children but does not return the node associated
            with p at level ``level-1``."""
            assert level <= self.highest_level
            if level in self.children:
                return self.children[level]
            else:
                return []

        def remove_connections(self):
            """
            Remove the connection between the highest-level node associated
            with p and that node's parent.
            """
            if self.highest_parent:
                q = self.highest_parent
                assert q.highest_level > self.highest_level
                assert self.highest_level in q.children
                assert self in q.children[self.level]
                q.children[self.level].remove(self)
                self.highest_parent = None

        def first_level_with_children(self):
            """Return the highest level at which a node associated with p
            has children."""
            try:
                return max(self.children.iterkeys())
            except ValueError:
                return _LOWEST_LEVEL

        def is_leaf_node(self, level):
            """Returns True if the node associated with p at level ``level``
            is a leaf node, i.e., it has no descendants other than nodes
            also associated with p."""
            assert level <= self.highest_level
            for (other_level, children) in self.children.iteritems():
                if other_level <= level and children:
                    return False
            else:
                return True

    def _build(self):
        """Build the cover tree using the Batch Construction algorithm
        from Beygelzimer, Kakade and Langford 2006."""

        def split_with_dist(dmax, pts_p_ds):
            """Split the points in a list into a those closer than dmax to p
            and those farther away.  Remove the far points from the original
            list.

            Parameters
            ----------
            dmax : float
                threshold distance
            pts_p_ds : list of (idx, dp) tuples
                A list of points (each with index idx) and their distance
                dp to a point p

            Return
            ------
            near_p_ds : list of (idx, dp) tuples
                List of points whose distance to p, dp, does not exceed dmax.
            far_p_ds : list of (idx, dp) tuples
                List of points whose distance to p, dp, exceeds dmax.

            Side effects
            ------------
            The elements in pts_p_ds with dp > dmax are removed.
            """
            near_p_ds = []
            far_p_ds = []

            new_pts_len = 0
            for i in xrange(len(pts_p_ds)):
                idx, dp = pts_p_ds[i]
                if dp <= dmax:
                    near_p_ds.append((idx, dp))
                elif dp <= 2*dmax:
                    far_p_ds.append((idx, dp))
                else:
                    pts_p_ds[new_pts_len] = pts_p_ds[i]
                    new_pts_len += 1
            pts_p_ds[:] = pts_p_ds[:new_pts_len]

            return near_p_ds, far_p_ds

        def split_without_dist(q_idx, dmax, pts_p_ds):
            """Split the points in a list into a those closer than dmax to q
            and those farther away.  Remove the far points from the original
            list.

            Parameters
            ----------
            q_idx : integer
                index of reference point q
            dmax : float
                threshold distance
            pts_p_ds : list of (idx, dp) tuples
                A list of points (each with index idx) and their distance
                dp to an unspecified point p

            Return
            ------
            near_q_ds : list of (idx, dq) tuples
                List of points whose distance to q, dq, does not exceed dmax.
            far_q_ds : list of (idx, dq) tuples
                List of points whose distance to q, dq, exceeds dmax.

            Side effects
            ------------
            The elements in pts_p_ds with dq > dmax are removed.
            """
            near_q_ds = []
            far_q_ds = []

            new_pts_len = 0
            for i in xrange(len(pts_p_ds)):
                idx, dp = pts_p_ds[i]
                dq = self.distance(self.data[q_idx], self.data[idx])
                if dq <= dmax:
                    near_q_ds.append((idx, dq))
                elif dq <= 2*dmax:
                    far_q_ds.append((idx, dq))
                else:
                    pts_p_ds[new_pts_len] = pts_p_ds[i]
                    new_pts_len += 1
            pts_p_ds[:] = pts_p_ds[:new_pts_len]

            return near_q_ds, far_q_ds

        global indent_level, max_indent_level
        indent_level = 0
        max_indent_level = 0
        def construct(p, near_p_ds, far_p_ds, i):
            """Main construction loop.

            Builds all of the descendants of the node associated with p at
            level i.  These include all of the points in near_p_ds, and may
            include some of the points in far_p_ds:

               x in near_p_ds  <=>     0 <= d(p,x) <= 2**i
               x in far_p_ds   <=>  2**i <  d(p,x) <  2**(i+1)
            
            Returns those points in far_p_ds that were not descendants
            of the node associated with p at level i
            """
            #assert all(d <= 2**i for (k,d) in near_p_ds)
            #assert all(2**i < d <= 2**(i+1) for (k,d) in far_p_ds)
            
            global indent_level, max_indent_level
            if False: #indent_level > max_indent_level:
                sys.stderr.write(
                    "%s construct(%d=%s, near=%s, far=%s, level=%d)\n"
                    % ('-'*indent_level, p.idx, str(self.data[p.idx]),
                       str([(k, d) for (k,d) in near_p_ds]),
                       str([(k, d) for (k,d) in far_p_ds]),
                       i) )
                max_indent_level = max(max_indent_level, indent_level)
                
            if not near_p_ds:
                return far_p_ds
            else:
                indent_level += 1
                
                nearer_p_ds, not_so_near_p_ds = split_with_dist(2**(i-1),
                                                                near_p_ds)
                near_p_ds = construct(p, nearer_p_ds, not_so_near_p_ds, i-1)
                
                # near_p_ds now contains points near to p at level i, but
                # not descendants of p at level i-1.
                # Make new children of p at level i from each one until
                # none remain
                while near_p_ds:
                    q_idx, _ = near_p_ds.pop()
                    q = CoverTree._Node(q_idx, i-1)
                    print ("%s Created node for pt %d at level %d, child of pt %d at level %d" %
                           ('-'*indent_level, q_idx, i-1, p.idx, i))
                    self.minlevel = min(i-1, self.minlevel)

                    #print ("%s Before split, p_idx=%d, near_p_ds=%s, far_p_ds=%s" %
                    #       ('-'*indent_level, p.idx,
                    #        str([(k, d) for (k,d) in near_p_ds]),
                    #        str([(k, d) for (k,d) in far_p_ds])))
                    
                    near_q_ds, far_q_ds = split_without_dist(
                        q_idx, 2**(i-1), near_p_ds)
                    near_q_ds2, far_q_ds2 = split_without_dist(
                        q_idx, 2**(i-1), far_p_ds)
                    near_q_ds.extend(near_q_ds2)
                    far_q_ds.extend(far_q_ds2)

                    #print ("%s After split, p_idx=%d, near_p_ds=%s, far_p_ds=%s" %
                    #       ('-'*indent_level, p.idx,
                    #        str([(k, d) for (k,d) in near_p_ds]),
                    #        str([(k, d) for (k,d) in far_p_ds])))
                    #print ("%s              q_idx=%d, near_q_ds=%s, far_q_ds=%s" %
                    #       ('-'*indent_level, q_idx,
                    #        str([(k, d) for (k,d) in near_q_ds]),
                    #        str([(k, d) for (k,d) in far_q_ds])))
                    
                    assert not (set(i for (i,d) in near_q_ds) &
                                set(i for (i,d) in far_q_ds))
                    assert not (set(i for (i,d) in near_q_ds+far_q_ds) &
                                set(i for (i,d) in far_p_ds))
                    unused_q_ds = construct(q, near_q_ds, far_q_ds, i-1)

                    p.add_child(q, i) # q_(i-1) is a child of p_i

                    # TODO: Figure out an effective way of not having
                    # to recalculate distances to p
                    new_near_p_ds, new_far_p_ds = split_without_dist(
                        p.idx, 2**i, unused_q_ds)
                    near_p_ds.extend(new_near_p_ds)
                    far_p_ds.extend(new_far_p_ds)
                    
                    #print ("%s Now, p_idx=%d, near_p_ds=%s, far_p_ds=%s" %
                    #       ('-'*indent_level, p.idx,
                    #        str([(k, d) for (k,d) in near_p_ds]),
                    #        str([(k, d) for (k,d) in far_p_ds])))

                indent_level -= 1
                return far_p_ds

        if self.n == 0:
            self.root = None
            self.maxlevel = self.minlevel = 0
        else:
            # Maximum distance between any two points can't exceed twice the
            # distance between the first point and any other point due to
            # the triangle inequality
            near_p_ds = [(j, self.distance(self.data[0], self.data[j]))
                         for j in np.arange(1, self.n)]
            far_p_ds = []
            try:
                maxdist = 2 * max(near_p_ds, key=operator.itemgetter(1))[1]
            except ValueError:
                maxdist = 1
            
            self.maxlevel = int(math.ceil(math.log(maxdist, 2)))+1
            self.minlevel = self.maxlevel

            p = CoverTree._Node(0, self.maxlevel)
            print "Created root node (pt %d) at level %d" % (0,self.maxlevel)
            unused_p_ds = construct(p, near_p_ds, far_p_ds, self.maxlevel)
            assert not unused_p_ds
            self.root = p

    def _get_children_dist(self, p, Q_p_ds, level):
        """Get the children of cover set Q at level ``level`` and the
        distances of them with point p.

        Parameters
        ----------
        p : array-like, shape pt_shape
            reference point p to measure distances
        Q_p_ds : list of (q, d(p,q)) tuples
            a list of all points in Q and their distance to p
        level : integer
            the level of the nodes in Q_p_ds
        
        Returns
        -------
        The children of Q and their distances to p.
        """
        
        # TODO: do this operation in place

        children = list(itertools.chain.from_iterable
                        (n.get_only_children(level) for n, _ in Q_p_ds))

        return Q_p_ds + list((q, self.distance(p, self.data[q.idx]))
                             for q in children)
        

    def _raw_query(self, p, bound):
        # The function bound(A), defined for a set of points p in A, receives
        # a list of tuples (q,d(p,q)) and returns a distance bound used to
        # cut off a nearest-neighbour search.  It should be a monotonic
        # function, i.e. bound(A) <= bound(B) whenever A is a superset of B
        #
        # __query returns a list of tuples Q = [(q,d(p,q))] of all points
        # satisfying d(p,q) <= bound(S), where S is the set of all points
        # in the cover tree.

        if self.root == None:
            return []

        Qi_p_ds = [(self.root, self.distance(p, self.data[self.root.idx]))]
        for i in xrange(self.maxlevel,self.minlevel-1,-1):
            # (a) Consider the set of children of Q_i
            #     [calculate their distances to x at the same time]
            Q_p_ds = self._get_children_dist(p, Qi_p_ds, i)

            # Refine bound
            d_p_Q = bound(Q_p_ds)

            # (b) Form next cover set
            # In-place filter trick from stackoverflow
            # (http://stackoverflow.com/questions/18418/elegant-way-to-remove-items-from-sequence-in-python)
            Qi_p_ds[:] = ((q, d) for q, d in Q_p_ds if d <= d_p_Q + 2**i)

        return Qi_p_ds
        
    def _query(self, p, k=1, eps=0, distance_upper_bound=np.inf):
        """Single-point query (internal)"""
        if eps != 0:
            raise NotImplementedError

        if k is None:
            def bound(Q_p_ds):
                return distance_upper_bound
        else:
            def bound(Q_p_ds):
                try:
                    return min(distance_upper_bound,
                               heapq.nsmallest(k, Q_p_ds,
                                               key=operator.itemgetter(1)
                                               )[-1][1])
                except ValueError:
                    return distance_upper_bound

        raw_result = self._raw_query(p, bound)
        if k:
            result = heapq.nsmallest(k, raw_result,
                                     key=operator.itemgetter(1))
        else:
            result = sorted(raw_result, key=operator.itemgetter(1))
        return [(d, q.idx) for (q, d) in result]
        

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
        x = np.asarray(x)
        if self.pt_shape:
            if np.shape(x)[-len(self.pt_shape):] != self.pt_shape:
                raise ValueError("x must consist of vectors of shape %s "
                                 "but has shape %s"
                                 % (self.pt_shape, np.shape(x)))
            retshape = np.shape(x)[:-len(self.pt_shape)]
        else:
            retshape = np.shape(x)

        if retshape:
            if k is None:
                dd = np.empty(retshape, dtype=np.object)
                ii = np.empty(retshape, dtype=np.object)
            elif k > 1:
                dd = np.empty(retshape + (k,), dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(retshape + (k,), dtype=np.int)
                ii.fill(self.n)
            elif k == 1:
                dd = np.empty(retshape, dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(retshape, dtype=np.int)
                ii.fill(self.n)
            else:
                raise ValueError("Requested %s nearest neighbors; acceptable numbers are integers greater than or equal to one, or None")
            for c in np.ndindex(retshape):
                hits = self._query(x[c], k=k, distance_upper_bound=distance_upper_bound)
                if k is None:
                    dd[c] = [d for (d,i) in hits]
                    ii[c] = [i for (d,i) in hits]
                elif k>1:
                    for j in range(len(hits)):
                        dd[c+(j,)], ii[c+(j,)] = hits[j]
                elif k==1:
                    if len(hits)>0:
                        dd[c], ii[c] = hits[0]
                    else:
                        dd[c] = np.inf
                        ii[c] = self.n
            return dd, ii
        else:
            hits = self._query(x, k=k, distance_upper_bound=distance_upper_bound)
            if k is None:
                return [d for (d,i) in hits], [i for (d,i) in hits]
            elif k==1:
                if len(hits)>0:
                    return hits[0]
                else:
                    return np.inf, self.n
            elif k>1:
                dd = np.empty(k,dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(k,dtype=np.int)
                ii.fill(self.n)
                for j in range(len(hits)):
                    dd[j], ii[j] = hits[j]
                return dd, ii
            else:
                raise ValueError("Requested %s nearest neighbors; acceptable numbers are integers greater than or equal to one, or None")
    
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
