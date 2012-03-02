# Copyright Patrick Varilly 2012
# Released under the scipy license
#
# Based on test_kdtree.py in scipy.spatial.tests, license as follows:
# 
# Copyright Anne M. Archibald 2008
# Released under the scipy license

from numpy.testing import assert_equal, assert_array_equal, assert_almost_equal, \
        assert_, run_module_suite

import numpy as np
from covertree import CoverTree, distance_matrix
from scipy.spatial.distance import sqeuclidean, cityblock, chebyshev

class ConsistencyTests:
    def test_nearest(self):
        x = self.x
        d, i = self.covertree.query(x, 1)
        assert_almost_equal(d, self.distance(x,self.data[i]))
        eps = 1e-8
        distance_to_x = np.vectorize(lambda y: self.distance(x,y))
        assert_(np.all(distance_to_x(self.data) > d**2-eps))

    def test_m_nearest(self):
        x = self.x
        m = self.m
        dd, ii = self.covertree.query(x, m)
        d = np.amax(dd)
        i = ii[np.argmax(dd)]
        assert_almost_equal(d, self.distance(x,self.data[i]))
        eps = 1e-8
        distance_to_x = np.vectorize(lambda y: self.distance(x,y))
        assert_equal(np.sum(distance_to_x(self.data) < d**2+eps), m)

    def test_points_near(self):
        x = self.x
        d = self.d
        dd, ii = self.covertree.query(x, k=self.covertree.n, distance_upper_bound=d)
        eps = 1e-8
        hits = 0
        for near_d, near_i in zip(dd,ii):
            if near_d==np.inf:
                continue
            hits += 1
            assert_almost_equal(near_d, self.distance(x, self.data[near_i]))
            assert_(near_d < d+eps,
                    "near_d=%g should be less than %g" % (near_d, d))
        distance_to_x = np.vectorize(lambda y: self.distance(x,y))
        assert_equal(np.sum(distance_to_x(self.data) < d**2+eps), hits)

    def test_approx(self):
        x = self.x
        k = self.k
        eps = 0.1
        d_real, i_real = self.covertree.query(x, k)
        d, i = self.covertree.query(x, k, eps=eps)
        assert_(np.all(d <= d_real*(1+eps)))


class test_random(ConsistencyTests):
    def setUp(self):
        self.n = 100
        self.m = 4
        self.data = np.random.randn(self.n, self.m)
        self.distance = sqeuclidean
        self.covertree = CoverTree(self.data, self.distance)#leafsize=2)
        self.x = np.random.randn(self.m)
        self.d = 0.2
        self.k = 10

class test_random_far(test_random):
    def setUp(self):
        test_random.setUp(self)
        self.x = np.random.randn(self.m) + 10

class test_small(ConsistencyTests):
    def setUp(self):
        self.data = np.array([[0,0,0],
                              [0,0,1],
                              [0,1,0],
                              [0,1,1],
                              [1,0,0],
                              [1,0,1],
                              [1,1,0],
                              [1,1,1]])
        self.distance = sqeuclidean
        self.covertree = CoverTree(self.data, self.distance)
        self.n = self.covertree.n
        self.m = self.covertree.m
        self.x = np.random.randn(3)
        self.d = 0.5
        self.k = 4

    def test_nearest(self):
        assert_array_equal(
                self.covertree.query((0,0,0.1), 1),
                (0.1,0))
        
    def test_nearest_two(self):
        assert_array_equal(
                self.covertree.query((0,0,0.1), 2),
                ([0.1,0.9],[0,1]))
        
#class test_small_nonleaf(test_small):
#    def setUp(self):
#        test_small.setUp(self)
#        self.covertree = KDTree(self.data,leafsize=1)
#
#class test_small_compiled(test_small):
#    def setUp(self):
#        test_small.setUp(self)
#        self.covertree = cKDTree(self.data)
#class test_small_nonleaf_compiled(test_small):
#    def setUp(self):
#        test_small.setUp(self)
#        self.covertree = cKDTree(self.data,leafsize=1)
#class test_random_compiled(test_random):
#    def setUp(self):
#        test_random.setUp(self)
#        self.covertree = cKDTree(self.data)
#class test_random_far_compiled(test_random_far):
#    def setUp(self):
#        test_random_far.setUp(self)
#        self.covertree = cKDTree(self.data)

class test_vectorization:
    def setUp(self):
        self.data = np.array([[0,0,0],
                              [0,0,1],
                              [0,1,0],
                              [0,1,1],
                              [1,0,0],
                              [1,0,1],
                              [1,1,0],
                              [1,1,1]])
        self.distance = sqeuclidean
        self.covertree = CoverTree(self.data,self.distance)

    def test_single_query(self):
        d, i = self.covertree.query(np.array([0,0,0]))
        assert_(isinstance(d,float))
        assert_(np.issubdtype(i, int))

    def test_vectorized_query(self):
        d, i = self.covertree.query(np.zeros((2,4,3)))
        assert_equal(np.shape(d),(2,4))
        assert_equal(np.shape(i),(2,4))

    def test_single_query_multiple_neighbors(self):
        s = 23
        kk = self.covertree.n+s
        d, i = self.covertree.query(np.array([0,0,0]),k=kk)
        assert_equal(np.shape(d),(kk,))
        assert_equal(np.shape(i),(kk,))
        assert_(np.all(~np.isfinite(d[-s:])))
        assert_(np.all(i[-s:]==self.covertree.n))

    def test_vectorized_query_multiple_neighbors(self):
        s = 23
        kk = self.covertree.n+s
        d, i = self.covertree.query(np.zeros((2,4,3)),k=kk)
        assert_equal(np.shape(d),(2,4,kk))
        assert_equal(np.shape(i),(2,4,kk))
        assert_(np.all(~np.isfinite(d[:,:,-s:])))
        assert_(np.all(i[:,:,-s:]==self.covertree.n))

    def test_single_query_all_neighbors(self):
        d, i = self.covertree.query([0,0,0],k=None,distance_upper_bound=1.1)
        assert_(isinstance(d,list))
        assert_(isinstance(i,list))

    def test_vectorized_query_all_neighbors(self):
        d, i = self.covertree.query(np.zeros((2,4,3)),k=None,distance_upper_bound=1.1)
        assert_equal(np.shape(d),(2,4))
        assert_equal(np.shape(i),(2,4))

        assert_(isinstance(d[0,0],list))
        assert_(isinstance(i[0,0],list))

#class test_vectorization_compiled:
#    def setUp(self):
#        self.data = np.array([[0,0,0],
#                              [0,0,1],
#                              [0,1,0],
#                              [0,1,1],
#                              [1,0,0],
#                              [1,0,1],
#                              [1,1,0],
#                              [1,1,1]])
#        self.covertree = cKDTree(self.data)
#
#    def test_single_query(self):
#        d, i = self.covertree.query([0,0,0])
#        assert_(isinstance(d,float))
#        assert_(isinstance(i,int))
#
#    def test_vectorized_query(self):
#        d, i = self.covertree.query(np.zeros((2,4,3)))
#        assert_equal(np.shape(d),(2,4))
#        assert_equal(np.shape(i),(2,4))
#
#    def test_vectorized_query_noncontiguous_values(self):
#        qs = np.random.randn(3,1000).T
#        ds, i_s = self.covertree.query(qs)
#        for q, d, i in zip(qs,ds,i_s):
#            assert_equal(self.covertree.query(q),(d,i))
#
#
#    def test_single_query_multiple_neighbors(self):
#        s = 23
#        kk = self.covertree.n+s
#        d, i = self.covertree.query([0,0,0],k=kk)
#        assert_equal(np.shape(d),(kk,))
#        assert_equal(np.shape(i),(kk,))
#        assert_(np.all(~np.isfinite(d[-s:])))
#        assert_(np.all(i[-s:]==self.covertree.n))
#
#    def test_vectorized_query_multiple_neighbors(self):
#        s = 23
#        kk = self.covertree.n+s
#        d, i = self.covertree.query(np.zeros((2,4,3)),k=kk)
#        assert_equal(np.shape(d),(2,4,kk))
#        assert_equal(np.shape(i),(2,4,kk))
#        assert_(np.all(~np.isfinite(d[:,:,-s:])))
#        assert_(np.all(i[:,:,-s:]==self.covertree.n))

class ball_consistency:

    def test_in_ball(self):
        l = self.T.query_ball_point(self.x, self.d, eps=self.eps)
        for i in l:
            assert_(self.distance(self.data[i], self.x) <= self.d*(1.+self.eps))

    def test_found_all(self):
        c = np.ones(self.T.n, dtype=np.bool)
        l = self.T.query_ball_point(self.x, self.d, eps=self.eps)
        c[l] = False
        assert_(np.all(self.distance(self.data[c], self.x) >=
                       self.d/(1.+self.eps)))

class test_random_ball(ball_consistency):

    def setUp(self,distance=sqeuclidean):
        n = 100
        m = 4
        self.data = np.random.randn(n,m)
        self.distance = distance
        self.T = CoverTree(self.data,self.distance)#,leafsize=2)
        self.x = np.random.randn(m)
        self.eps = 0
        self.d = 0.2

class test_random_ball_approx(test_random_ball):

    def setUp(self):
        test_random_ball.setUp(self)
        self.eps = 0.1

class test_random_ball_far(test_random_ball):

    def setUp(self):
        test_random_ball.setUp(self)
        self.d = 2.

class test_random_ball_l1(test_random_ball):

    def setUp(self):
        test_random_ball.setUp(self,distance=cityblock)

class test_random_ball_linf(test_random_ball):

    def setUp(self):
        test_random_ball.setUp(self,distance=chebyshev)

def test_random_ball_vectorized():

    n = 20
    m = 5
    T = CoverTree(np.random.randn(n,m), distance=sqeuclidean)

    r = T.query_ball_point(np.random.randn(2,3,m),1)
    assert_equal(r.shape,(2,3))
    assert_(isinstance(r[0,0],list))

class two_trees_consistency:

    def test_all_in_ball(self):
        r = self.T1.query_ball_tree(self.T2, self.d, eps=self.eps)
        for i, l in enumerate(r):
            for j in l:
                assert_(self.distance(self.data1[i],self.data2[j]) <=
                        self.d*(1.+self.eps))
    def test_found_all(self):
        r = self.T1.query_ball_tree(self.T2, self.d, eps=self.eps)
        for i, l in enumerate(r):
            c = np.ones(self.T2.n,dtype=np.bool)
            c[l] = False
            assert_(np.all(self.distance(self.data2[c],self.data1[i]) >=
                           self.d/(1.+self.eps)))

class test_two_random_trees(two_trees_consistency):

    def setUp(self, distance=sqeuclidean):
        n = 50
        m = 4
        self.data1 = np.random.randn(n,m)
        self.distance = distance
        self.T1 = CoverTree(self.data1,self.distance)#leafsize=2)
        self.data2 = np.random.randn(n,m)
        self.T2 = CoverTree(self.data2,self.distance)#leafsize=2)
        self.eps = 0
        self.d = 0.2

class test_two_random_trees_far(test_two_random_trees):

    def setUp(self):
        test_two_random_trees.setUp(self)
        self.d = 2

class test_two_random_trees_linf(test_two_random_trees):

    def setUp(self):
        test_two_random_trees.setUp(self, distance=chebyshev)


def test_distance_l2():
    assert_almost_equal(sqeuclidean([0,0],[1,1]),2)
def test_distance_l1():
    assert_almost_equal(cityblock([0,0],[1,1]),2)
def test_distance_linf():
    assert_almost_equal(chebyshev([0,0],[1,1]),1)
#def test_distance_vectorization():
#    x = np.random.randn(10,1,3)
#    y = np.random.randn(1,7,3)
#    assert_equal(distance(x,y).shape,(10,7))

class test_count_neighbors:

    def setUp(self):
        n = 50
        m = 2
        self.T1 = CoverTree(np.random.randn(n,m),distance=sqeuclidean)#leafsize=2)
        self.T2 = CoverTree(np.random.randn(n,m),distance=sqeuclidean)#,leafsize=2)

    def test_one_radius(self):
        r = 0.2
        assert_equal(self.T1.count_neighbors(self.T2, r),
                np.sum([len(l) for l in self.T1.query_ball_tree(self.T2,r)]))

    def test_large_radius(self):
        r = 1000
        assert_equal(self.T1.count_neighbors(self.T2, r),
                np.sum([len(l) for l in self.T1.query_ball_tree(self.T2,r)]))

    def test_multiple_radius(self):
        rs = np.exp(np.linspace(np.log(0.01),np.log(10),3))
        results = self.T1.count_neighbors(self.T2, rs)
        assert_(np.all(np.diff(results)>=0))
        for r,result in zip(rs, results):
            assert_equal(self.T1.count_neighbors(self.T2, r), result)

class test_sparse_distance_matrix:
    def setUp(self):
        n = 50
        m = 4
        self.distance = sqeuclidean
        self.T1 = CoverTree(np.random.randn(n,m),self.distance)#,leafsize=2)
        self.T2 = CoverTree(np.random.randn(n,m),self.distance)#,leafsize=2)
        self.r = 0.3

    def test_consistency_with_neighbors(self):
        M = self.T1.sparse_distance_matrix(self.T2, self.r)
        r = self.T1.query_ball_tree(self.T2, self.r)
        for i,l in enumerate(r):
            for j in l:
                assert_equal(M[i,j],self.distance(self.T1.data[i],self.T2.data[j]))
        for ((i,j),d) in M.items():
            assert_(j in r[i])

    def test_zero_distance(self):
        M = self.T1.sparse_distance_matrix(self.T1, self.r)

def test_distance_matrix():
    m = 10
    n = 11
    k = 4
    xs = np.random.randn(m,k)
    ys = np.random.randn(n,k)
    distance = sqeuclidean
    ds = distance_matrix(xs,ys,distance)
    assert_equal(ds.shape, (m,n))
    for i in range(m):
        for j in range(n):
            assert_almost_equal(distance(xs[i],ys[j]),ds[i,j])
#def test_distance_matrix_looping():
#    m = 10
#    n = 11
#    k = 4
#    xs = np.random.randn(m,k)
#    ys = np.random.randn(n,k)
#    ds = distance_matrix(xs,ys)
#    dsl = distance_matrix(xs,ys,threshold=1)
#    assert_equal(ds,dsl)

def check_onetree_query(T,d):
    r = T.query_ball_tree(T, d)
    s = set()
    for i, l in enumerate(r):
        for j in l:
            if i<j:
                s.add((i,j))

    assert_(s == T.query_pairs(d))

def test_onetree_query():
    np.random.seed(0)
    n = 100
    k = 4
    points = np.random.randn(n,k)
    distance = sqeuclidean
    T = CoverTree(points,distance)
    yield check_onetree_query, T, 0.1

    points = np.random.randn(3*n,k)
    points[:n] *= 0.001
    points[n:2*n] += 2
    T = CoverTree(points,distance)
    yield check_onetree_query, T, 0.1
    yield check_onetree_query, T, 0.001
    yield check_onetree_query, T, 0.00001
    yield check_onetree_query, T, 1e-6

def test_query_pairs_single_node():
    distance = sqeuclidean
    tree = CoverTree([[0, 1]],distance)
    assert_equal(tree.query_pairs(0.5), set())


def test_ball_point_ints():
    """Description from test_kdtree.py: Regression test for #1373."""
    x, y = np.mgrid[0:4, 0:4]
    points = zip(x.ravel(), y.ravel())
    distance = sqeuclidean
    tree = CoverTree(points,distance)
    assert_equal([4, 8, 9, 12], tree.query_ball_point((2, 0), 1))
    points = np.asarray(points, dtype=np.float)
    tree = CoverTree(points,distance)
    assert_equal([4, 8, 9, 12], tree.query_ball_point((2, 0), 1))


if __name__=="__main__":
    run_module_suite()
