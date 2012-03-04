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

from __future__ import division

import numpy as np
from collections import defaultdict
import operator
import math
import itertools
import sys
from heapq import heappush, heappop
import random
import scipy.sparse

__all__ = ['CoverTree', 'distance_matrix']

class CoverTree(object):
    """
    Cover tree for quick nearest-neighbor lookup in general metric spaces.

    TODO: Write more
    """

    # A node at level i can have immediate children within a distance d_i =
    # child_d[i] and descendants within a distance D_i = heir_d[i].
    # Strictly speaking, the only requirement for using a cover tree is that
    #
    #    D_i = d_i + d_(i-1) + ...
    #
    # be defined, but the construction algorithm used here (batch
    # construction in Beygelzimer, Kakade and Langford) only works when
    #
    #    ... d_(i-1) < d_i < d_(i+1) < ...
    #
    # A convenient choice is d_i = b**i, with b > 1, whereby
    # D_i = b**i + b**(i-1) + ... = (b/(b-1)) * d_i
    #
    # Below, I implement these two fundamental scales as a lazy dictionary
    class _lazy_child_dist(dict):
        def __init__(self, base, *a, **kw):
            dict.__init__(self, *a, **kw)
            self.b = base
            
        def __missing__(self, i):
            self[i] = value = self.b ** i
            return value

    class _lazy_heir_dist(dict):
        def __init__(self, base, *a, **kw):
            dict.__init__(self, *a, **kw)
            self.b = base
            
        def __missing__(self, i):
            self[i] = value = self.b ** (i+1) / (self.b - 1)
            return value

    def __init__(self, data, distance, leafsize=10, base=2):
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
        leafsize : positive int
            The number of points at which the algorithm switches over to
            brute-force.
        base : positive int
            The factor by which the radius of nodes at level i-1 shrinks
            with respect to nodes at level i

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
        self.leafsize = leafsize
        if self.leafsize < 1:
            raise ValueError("leafsize must be at least 1")

        self._child_d = CoverTree._lazy_child_dist(base)
        self._heir_d = CoverTree._lazy_heir_dist(base)

        self.tree = self._build()

    class _Node(object):
        """
        A node in the cover tree.

        In the implicit representation, each node in the tree has a
        fixed level i, an associated point p and a list of children
        in level i-1.  If a point p first appears at level i, then
        a node corresponding to p appears at every lower level.

        In the explicit representation used here, we only keep track of the
        nodes p_i that have nontrivial children.  Furthermore, we also
        use leaf nodes (like KDTree) to group together small numbers of
        nearby points at the lower levels.
        """
        pass

    class _InnerNode(_Node):
        # children are within _d[level] of data[ctr_idx]
        # descendants are within _D[level]
        # ctr_idx is one integer
        def __init__(self, ctr_idx, level, radius, children):
            self.ctr_idx = ctr_idx
            self.level = level
            self.radius = radius
            self.children = children
            self.num_children = sum(c.num_children for c in children)

        def __repr__(self):
            return ("<_InnerNode: ctr_idx=%d, level=%d (radius=%f), "
                    "len(children)=%d, num_children=%d>" %
                    (self.ctr_idx, self.level,
                     self.radius, len(self.children), self.num_children))
        
    class _LeafNode(_Node):
        # idx is an array of integers
        def __init__(self, idx, ctr_idx, radius):
            self.idx = idx
            self.ctr_idx = ctr_idx
            self.radius = radius
            self.num_children = len(idx)

        def __repr__(self):
            return('_LeafNode(idx=%s, ctr_idx=%d, radius=%f)' %
                   (repr(self.idx), self.ctr_idx, self.radius))
        

    def _build(self):
        """Build the cover tree using the Batch Construction algorithm
        from Beygelzimer, Kakade and Langford 2006."""

        child_d = self._child_d
        heir_d = self._heir_d

        def split_with_dist(dmax, Dmax, pts_p_ds):
            """Split the points in a list into a those closer than dmax to p
            and those up to Dmax away.  Remove the far points from the
            original list, preserve those closer than Dmax.

            Parameters
            ----------
            dmax : float
                inner threshold distance
            Dmax : float
                outer threshold distance
            pts_p_ds : list of (idx, dp) tuples
                A list of points (each with index idx) and their distance
                dp to a point p

            Return
            ------
            near_p_ds : list of (idx, dp) tuples
                List of points whose distance to p, dp, satisfies
                0 <= dp <= dmax
            far_p_ds : list of (idx, dp) tuples
                List of points whose distance to p, dp, satisfies
                dmax < dp <= Dmax

            Side effects
            ------------
            The elements in pts_p_ds with dp < Dmax are removed.
            """
            near_p_ds = []
            far_p_ds = []

            new_pts_len = 0
            for i in xrange(len(pts_p_ds)):
                idx, dp = pts_p_ds[i]
                if dp <= dmax:
                    near_p_ds.append((idx, dp))
                elif dp <= Dmax:
                    far_p_ds.append((idx, dp))
                else:
                    pts_p_ds[new_pts_len] = pts_p_ds[i]
                    new_pts_len += 1
            pts_p_ds[:] = pts_p_ds[:new_pts_len]

            return near_p_ds, far_p_ds

        def split_without_dist(q_idx, dmax, Dmax, pts_p_ds):
            """Split the points in a list into a those closer than dmax to q
            and, those up to Dmax away, and those beyond.  Remove the far
            points from the original list, preserve those closer than Dmax.

            Parameters
            ----------
            q_idx : integer
                index of reference point q
            dmax : float
                inner threshold distance
            Dmax : float
                outer threshold distance
            pts_p_ds : list of (idx, dp) tuples
                A list of points (each with index idx) and their distance
                dp to an unspecified point p

            Return
            ------
            near_q_ds : list of (idx, dq) tuples
                List of points whose distance to q, dq, satisfies
                0 <= dq <= dmax
            far_q_ds : list of (idx, dq) tuples
                List of points whose distance to q, dq, satisfies
                dmax < dq <= Dmax

            Side effects
            ------------
            The elements in pts_p_ds with dq < Dmax are removed.
            """
            near_q_ds = []
            far_q_ds = []

            new_pts_len = 0
            for i in xrange(len(pts_p_ds)):
                idx, dp = pts_p_ds[i]
                dq = self.distance(self.data[q_idx], self.data[idx])
                if dq <= dmax:
                    near_q_ds.append((idx, dq))
                elif dq <= Dmax:
                    far_q_ds.append((idx, dq))
                else:
                    pts_p_ds[new_pts_len] = pts_p_ds[i]
                    new_pts_len += 1
            pts_p_ds[:] = pts_p_ds[:new_pts_len]

            return near_q_ds, far_q_ds

        def construct(p_idx, near_p_ds, far_p_ds, i):
            """Main construction loop.

            Builds all of the descendants of the node associated with p at
            level i.  These include all of the points in near_p_ds, and may
            include some of the points in far_p_ds:

               x in near_p_ds  <=>     0 <= d(p,x) <= d_i
               x in far_p_ds   <=>   d_i <  d(p,x) <  d_(i+1)
            
            Returns those points in far_p_ds that were not descendants
            of the node associated with p at level i
            """
            assert all(d <= child_d[i] for (k,d) in near_p_ds)
            assert all(child_d[i] < d <= child_d[i+1] for (k,d) in far_p_ds)
            
            if len(near_p_ds) + len(far_p_ds) <= self.leafsize:
                idx = [ii for (ii, d) in itertools.chain(near_p_ds,
                                                         far_p_ds)]
                radius = max(d for (ii, d) in itertools.chain(near_p_ds,
                                                              far_p_ds,
                                                              [(0.0, None)]))
                #print("Building level %d leaf node for p_idx=%d with %s"
                #      % (i, p_idx, str(idx)))
                node = CoverTree._LeafNode(idx, p_idx, radius)
                return node, []
            else:
                # Remove points very near to p, and as many as possible of
                # those that are just "near"
                nearer_p_ds, so_so_near_p_ds = split_with_dist(
                    child_d[i-1], child_d[i], near_p_ds)
                p_im1, near_p_ds = construct(p_idx, nearer_p_ds,
                                             so_so_near_p_ds, i-1)

                # If no near points remain, p_i would only have the
                # trivial child p_im1.  Skip directly to p_im1 in the
                # explicit representation
                if not near_p_ds:
                    #print("Passing though level %d child node %s up to level %d" %
                    #      (i-1, str(p_im1), i))
                    return p_im1, far_p_ds
                else:
                    # near_p_ds now contains points near to p at level i,
                    # but not descendants of p at level i-1.
                    #
                    # Make new children of p at level i from each one until
                    # none remain
                    children = [p_im1]
                    while near_p_ds:
                        q_idx, _ = random.choice(near_p_ds)

                        near_q_ds, far_q_ds = split_without_dist(
                            q_idx, child_d[i-1], child_d[i], near_p_ds)
                        near_q_ds2, far_q_ds2 = split_without_dist(
                            q_idx, child_d[i-1], child_d[i], far_p_ds)
                        near_q_ds += near_q_ds2
                        far_q_ds += far_q_ds2
                        
                        #assert not (set(i for (i,d) in near_q_ds) &
                        #            set(i for (i,d) in far_q_ds))
                        #assert not (set(i for (i,d) in near_q_ds+far_q_ds) &
                        #            set(i for (i,d) in far_p_ds))
                    
                        q_im1, unused_q_ds = construct(
                            q_idx, near_q_ds, far_q_ds, i-1)
                        
                        children.append(q_im1)
                        
                        # TODO: Figure out an effective way of not having
                        # to recalculate distances to p
                        new_near_p_ds, new_far_p_ds = split_without_dist(
                            p_idx, child_d[i], child_d[i+1], unused_q_ds)
                        near_p_ds += new_near_p_ds
                        far_p_ds += new_far_p_ds

                    p_i = CoverTree._InnerNode(p_idx, i, heir_d[i], children)
                    #print("Creating level %d inner node with %d children, "
                    #      "remaining points = %s" %
                    #      (i, len(p_i.children), str(far_p_ds)))
                    return p_i, far_p_ds

        if self.n == 0:
            self.root = CoverTree._LeafNode(idx=[], ctr_idx=-1, radius=0)
        else:
            # Maximum distance between any two points can't exceed twice the
            # distance between some fixed point and any other point due to
            # the triangle inequality
            p_idx = random.randrange(self.n)
            near_p_ds = [(j, self.distance(self.data[p_idx], self.data[j]))
                         for j in np.arange(self.n)]
            far_p_ds = []
            try:
                maxdist = 2 * max(near_p_ds, key=operator.itemgetter(1))[1]
            except ValueError:
                maxdist = 1

            # We'll place p at a level such that all other points
            # are "near" p, in the sense of construct() above
            maxlevel = 0
            while maxdist > child_d[maxlevel]:
                maxlevel += 1

            self.root, unused_p_ds = construct(p_idx, near_p_ds,
                                               far_p_ds, maxlevel)
            assert not unused_p_ds

    def _query(self, p, k=1, eps=0, distance_upper_bound=np.inf):
        if not self.root:
            return []

        dist_to_ctr = self.distance(p, self.data[self.root.ctr_idx])
        min_distance = max(0.0, dist_to_ctr - self.root.radius)
        
        # priority queue for chasing nodes
        # entries are:
        #  minimum distance between the node area and the target
        #  distance between node center and target
        #  the node
        q = [(min_distance,
              dist_to_ctr,
              self.root)]
        # priority queue for the nearest neighbors
        # furthest known neighbor first
        # entries are (-distance, i)
        neighbors = []

        if eps==0:
            epsfac=1
        else:
            epsfac = 1/(1+eps)
        
        while q:
            min_distance, dist_to_ctr, node = heappop(q)
            if isinstance(node, CoverTree._LeafNode):
                # brute-force
                for i in node.idx:
                    if i == node.ctr_idx:
                        d = dist_to_ctr
                    else:
                        d = self.distance(p, self.data[i])
                    if d <= distance_upper_bound:
                        if len(neighbors) == k:
                            heappop(neighbors)
                        heappush(neighbors, (-d, i))
                        if len(neighbors) == k:
                            distance_upper_bound = -neighbors[0][0]
            else:
                # we don't push nodes that are too far onto the queue at
                # all, but since the distance_upper_bound decreases, we
                # might get here even if the cell's too far
                if min_distance > distance_upper_bound*epsfac:
                    # since this is the nearest node, we're done, bail out
                    break

                for child in node.children:
                    if child.ctr_idx == node.ctr_idx:
                        d = dist_to_ctr
                    else:
                        d = self.distance(p, self.data[child.ctr_idx])
                    min_distance = max(0.0, d - child.radius)
                    
                    # child might be too far, if so, don't bother pushing it
                    if min_distance <= distance_upper_bound*epsfac:
                        heappush(q, (min_distance, d, child))

        return sorted([(-d,i) for (d,i) in neighbors])
    
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
                hits = self._query(x[c], k=k, eps=eps, distance_upper_bound=distance_upper_bound)
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
            hits = self._query(x, k=k, eps=eps, distance_upper_bound=distance_upper_bound)
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
    
    def _query_ball_point(self, x, r, eps=0):
        def traverse_checking(node):
            d_x_node = self.distance(x, self.data[node.ctr_idx])
            min_distance = max(0.0, d_x_node - node.radius)
            max_distance = d_x_node + node.radius
            if min_distance > r / (1. + eps):
                return []
            elif max_distance < r * (1. + eps):
                return traverse_no_checking(node)
            elif isinstance(node, CoverTree._LeafNode):
                return list(i for i in node.idx
                            if self.distance(x, self.data[i]) <= r)
            else:
                return list(itertools.chain.from_iterable(
                        traverse_checking(child)
                        for child in node.children))

        def traverse_no_checking(node):
            if isinstance(node, CoverTree._LeafNode):
                return node.idx
            else:
                return list(itertools.chain.from_iterable(
                        traverse_no_checking(child)
                        for child in node.children))

        return traverse_checking(self.root)

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
        x = np.asarray(x)
        if self.pt_shape and x.shape[-len(self.pt_shape):] != self.pt_shape:
            raise ValueError("Searching for a point of shape %s in a " \
                             "CoverTree with points of shape %s" %
                             (x.shape[-len(self.pt_shape):],
                              self.pt_shape))
                             
        if len(x.shape) == 1:
            return self._query_ball_point(x, r, eps)
        else:
            if self.pt_shape:
                retshape = x.shape[:-len(self.pt_shape)]
            else:
                retshape = x.shape
            result = np.empty(retshape, dtype=np.object)
            for c in np.ndindex(retshape):
                result[c] = self._query_ball_point(x[c], r, eps=eps)
            return result
    
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
        results = [[] for i in range(self.n)]
        def traverse_checking(node1, node2):
            d = self.distance(self.data[node1.ctr_idx],
                              other.data[node2.ctr_idx])
            min_distance = max(0.0, d - node1.radius - node2.radius)
            max_distance = d + node1.radius + node2.radius
            if min_distance > r/(1.+eps):
                return
            elif max_distance < r*(1.+eps):
                traverse_no_checking(node1, node2)
            elif isinstance(node1, CoverTree._LeafNode):
                if isinstance(node2, CoverTree._LeafNode):
                    for i in node1.idx:
                        results[i] += list(
                            j for j in node2.idx
                            if self.distance(other.data[j], self.data[i]) <= r)
                else:
                    for child2 in node2.children:
                        traverse_checking(node1, child2)
            elif isinstance(node2, CoverTree._LeafNode):
                for child1 in node1.children:
                    traverse_checking(child1, node2)
            else:
                # Break down bigger node
                if node1.radius > node2.radius:
                    for child1 in node1.children:
                        traverse_checking(child1, node2)
                else:
                    for child2 in node2.children:
                        traverse_checking(node1, child2)

        def traverse_no_checking(node1, node2):
            if isinstance(node1, CoverTree._LeafNode):
                if isinstance(node2, CoverTree._LeafNode):
                    for i in node1.idx:
                        results[i] += node2.idx
                else:
                    for child2 in node2.children:
                        traverse_no_checking(node1, child2)
            else:
                for child1 in node1.children:
                    traverse_checking(child1, node2)

        traverse_checking(self.root, other.root)
        return results

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
            set of pairs (i,j), i<j, for which the corresponding positions
            are close.
        """
        results = set()
        visited = set()
        def test_set_visited(node1, node2):
            i, j = sorted((id(node1),id(node2)))
            if (i,j) in visited:
                return True
            else:
                visited.add((i,j))
                return False
        def traverse_checking(node1, node2):
            if test_set_visited(node1, node2):
                return

            if id(node2)<id(node1):
                # This node pair will be visited in the other order
                #return
                pass

            if isinstance(node1, CoverTree._LeafNode):
                if isinstance(node2, CoverTree._LeafNode):
                    for i in node1.idx:
                        for j in node2.idx:
                            if self.distance(self.data[i], self.data[j]) <= r:
                                if i < j:
                                    results.add((i,j))
                                elif j < i:
                                    results.add((j,i))
                else:
                    for child2 in node2.children:
                        traverse_checking(node1, child2)
            elif isinstance(node2, CoverTree._LeafNode):
                for child1 in node1.children:
                    traverse_checking(child1, node2)
            else:
                d_1_2 = self.distance(self.data[node1.ctr_idx],
                                      self.data[node2.ctr_idx])
                min_distance = d_1_2 - node1.radius - node2.radius
                max_distance = d_1_2 + node1.radius + node2.radius
                if min_distance > r/(1.+eps):
                    return
                elif max_distance < r*(1.+eps):
                    for child1 in node1.children:
                        traverse_no_checking(child1, node2)
                else:
                    # Break down bigger node
                    if node1.radius > node2.radius:
                        for child1 in node1.children:
                            traverse_checking(child1, node2)
                    else:
                        for child2 in node2.children:
                            traverse_checking(node1, child2)

        def traverse_no_checking(node1, node2):
            if test_set_visited(node1, node2):
                return

            if id(node2)<id(node1):
                # This node pair will be visited in the other order
                #return
                pass
            if isinstance(node1, CoverTree._LeafNode):
                if isinstance(node2, CoverTree._LeafNode):
                    for i in node1.idx:
                        for j in node2.idx:
                            if i<j:
                                results.add((i,j))
                            elif j<i:
                                results.add((j,i))
                else:
                    for child2 in node2.children:
                        traverse_no_checking(node1, child2)
            else:
                for child1 in node1.children:
                    traverse_no_checking(child1, node2)

        traverse_checking(self.root, self.root)
        return results
    
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

        def traverse(node1, node2, idx):
            d_1_2 = self.distance(self.data[node1.ctr_idx],
                                  other.data[node2.ctr_idx])
            min_r = d_1_2 - node1.radius - node2.radius
            max_r = d_1_2 + node1.radius + node2.radius
            c_greater = r[idx] > max_r
            result[idx[c_greater]] += node1.num_children * node2.num_children
            idx = idx[(min_r <= r[idx]) & (r[idx] <= max_r)]
            if len(idx)==0:
                return

            if isinstance(node1,CoverTree._LeafNode):
                if isinstance(node2,CoverTree._LeafNode):
                    ds = [self.distance(self.data[i], other.data[j])
                          for i in node1.idx
                          for j in node2.idx]
                    ds.sort()
                    result[idx] += np.searchsorted(ds, r[idx], side='right')
                else:
                    for child2 in node2.children:
                        traverse(node1, child2, idx)
            elif isinstance(node2, CoverTree._LeafNode):
                for child1 in node1.children:
                    traverse(child1, node2, idx)
            else:
                # Break down bigger node
                if node1.radius > node2.radius:
                    for child1 in node1.children:
                        traverse(child1, node2, idx)
                else:
                    for child2 in node2.children:
                        traverse(node1, child2, idx)

        if np.shape(r) == ():
            r = np.array([r])
            result = np.zeros(1,dtype=int)
            traverse(self.root, other.root, np.arange(1))
            return result[0]
        elif len(np.shape(r))==1:
            r = np.asarray(r)
            n, = r.shape
            result = np.zeros(n,dtype=int)
            traverse(self.root, other.root, np.arange(n))
            return result
        else:
            raise ValueError("r must be either a single value or a one-dimensional array of values")

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
        result = scipy.sparse.dok_matrix((self.n,other.n))
        
        def traverse(node1, node2):
            d_1_2 = self.distance(self.data[node1.ctr_idx],
                                  other.data[node2.ctr_idx])
            min_distance_1_2 = d_1_2 - node1.radius - node2.radius
            if min_distance_1_2 > max_distance:
                return
            elif isinstance(node1, CoverTree._LeafNode):
                if isinstance(node2, CoverTree._LeafNode):
                    for i in node1.idx:
                        for j in node2.idx:
                            d = self.distance(self.data[i],
                                              other.data[j])
                            if d <= max_distance:
                                result[i,j] = d
                else:
                    for child2 in node2.children:
                        traverse(node1, child2)
            elif isinstance(node2, CoverTree._LeafNode):
                for child1 in node1.children:
                    traverse(child1, node2)
            else:
                # Break down bigger node
                if node1.radius > node2.radius:
                    for child1 in node1.children:
                        traverse(child1, node2)
                else:
                    for child2 in node2.children:
                        traverse(node1, child2)
                
        traverse(self.root, other.root)

        return result

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
