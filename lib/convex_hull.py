"""
Library functions for graham algorithm implementation (find the convex hull
of a given list of points).
"""

import copy
import numpy as np


# Define angle and vectors operations
def vector(A, B):
    """
    Define a vector from the inputed points A, B.
    """
    return (B[0] - A[0], B[1] - A[1])


def determinant(u, v):
    """
    Compute the determinant of 2 vectors.
    --------
    Inputs:
    u, v : vectors (list of length 2)
        Vectors for which the determinant must be computed.
    ----------
    Returns:
    determinant : real
        Determinant of the input vectors.
    """
    return u[0] * v[1] - u[1] * v[0]


def pseudo_angle(A, B, C):
    """
    Define a pseudo-angle from the inputed points A, B, C.
    """
    u = vector(A, B)
    v = vector(A, C)
    return determinant(u, v)


def distance(A, B):
    """
    Compute the euclidian distance betwwen 2 points of the plan.
    ----------
    Inputs:
    A, B : points (list of length 2)
        Points between which the distance must be computed.
    ----------
    Returns:
    distance : real
        Euclidian distance between A, B.
    """
    x, y = vector(A, B)
    return np.sqrt(x ** 2 + y ** 2)


# Define lexicographic and composition order
def lexico(A, B):
    """
    Define the lexicographic order between points A, B.
    A <= B if:
        - y_a < y_b
    or  - y_a = y_b and x_a < x_b
    ----------
    Inputs:
    A, B : points (list of length 2)
        Points compared for the lexicographic order
    ----------
    Returns:
    lexico : boolean
        True if A<=B, False otherwise.
    """
    return (A[1] < B[1]) or (A[1] == B[1] and A[0] <= B[0])


def min_lexico(s):
    """
    Return the minimum of a list of points for the lexicographic order.
    ----------
    Inputs:
    s : list of points
        List of points in which we look for the minimum for the lexicographic
        order
    ----------
    Returns:
    m : point (list of length 2)
        Minimum for the lexicographic order in the input list s.
    """
    m = s[0]
    for x in s:
        if lexico(x, m): m = x
    return m


def comp(Omega, A, B):
    """
    Test the order relation between points such that A <= B :
    This is True if :
        - (Omega-A,Omega-B) is a right-hand basis of the plan.
    or  - (Omega-A,Omega-B) are linearly dependent.
    ----------
    Inputs:
    Omega, A, B : points (list of length 2)
        Points compared for the composition order
    ----------
    Returns:
    comp : boolean
        True if A <= B, False otherwise.
    """
    t = pseudo_angle(Omega, A, B)
    if t > 0:
        return True
    elif t < 0:
        return False
    else:
        return distance(Omega, A) <= distance(Omega, B)


# Implement quicksort
def partition(s, l, r, order):
    """
    Take a random element of a list 's' between indexes 'l', 'r' and place it
    at its right spot using relation order 'order'. Return the index at which
    it was placed.
    ----------
    Inputs:
    s : list
        List of elements to be ordered.
    l, r : int
        Index of the first and last elements to be considered.
    order : func: A, B -> bool
        Relation order between 2 elements A, B that returns True if A<=B,
        False otherwise.
    ----------
    Returns:
    index : int
        Index at which have been placed the element chosen by the function.
    """
    i = l - 1
    for j in range(l, r):
        if order(s[j], s[r]):
            i = i + 1
            temp = copy.deepcopy(s[i])
            s[i] = copy.deepcopy(s[j])
            s[j] = copy.deepcopy(temp)
    temp = copy.deepcopy(s[i+1])
    s[i+1] = copy.deepcopy(s[r])
    s[r] = copy.deepcopy(temp)
    return i + 1


def sort_aux(s, l, r, order):
    """
    Sort a list 's' between indexes 'l', 'r' using relation order 'order' by
    dividing it in 2 sub-lists and sorting these.
    """
    if l <= r:
        # Call partition function that gives an index on which the list will be
        #divided
        q = partition(s, l, r, order)
        sort_aux(s, l, q - 1, order)
        sort_aux(s, q + 1, r, order)


def quicksort(s, order):
    """
    Implement the quicksort algorithm on the list 's' using the relation order
    'order'.
    """
    # Call auxiliary sort on whole list.
    sort_aux(s, 0, len(s) - 1, order)


def sort_angles_distances(Omega, s):
    """
    Sort the list of points 's' for the composition order given reference point
    Omega.
    """
    order = lambda A, B: comp(Omega, A, B)
    quicksort(s, order)


# Define fuction for stacks (use here python lists with stack operations).
def empty_stack(): return []
def stack(S, A): S.append(A)
def unstack(S): S.pop()
def stack_top(S): return S[-1]
def stack_sub_top(S): return S[-2]


# Alignement handling
def del_aux(s, k, Omega):
    """
    Returne the index of the first element in 's' after the index 'k' that is
    not linearly dependent of (Omega-s[k]), first element not aligned to Omega
    and s[k].
    ----------
    Inputs:
    s : list
        List of points
    k : int
        Index from which to look for aligned points in s.
    Omega : point (list of length 2)
        Reference point in 's' to test alignement.
    ----------
    Returns:
    index : int
        Index of the first element not aligned to Omega and s[k].
    """
    p = s[k]
    j = k + 1
    while (j < len(s)) and (pseudo_angle(Omega, p, s[j]) == 0):
        j = j + 1
    return j


def del_aligned(s, Omega):
    """
    Deprive a list of points 's' from all intermediary points that are aligned
    using Omega as a reference point.
    ----------
    Inputs:
    s : list
        List of points
    Omega : point (list of length 2)
        Point of reference to test alignement.
    -----------
    Returns:
    s1 : list
        List of points where no points are aligned using reference point Omega.
    """
    s1 = [s[0]]
    n = len(s)
    k = 1
    while k < n:
        # Get the index of the last point aligned with (Omega-s[k])
        k = del_aux(s, k, Omega)
        s1.append(s[k - 1])
    return s1


# Implement Graham algorithm
def convex_hull(A):
    """
    Implement Graham algorithm to find the convex hull of a given list of
    points, handling aligned points.
    ----------
    Inputs:
    A : list
        List of points for which the the convex hull must be returned.
    ----------
    S : list
        List of points defining the convex hull of the input list A.
    """
    S = empty_stack()
    Omega = min_lexico(A)
    sort_angles_distances(Omega, A)
    A = del_aligned(A, Omega)
    stack(S, A[0])
    stack(S, A[1])
    stack(S, A[2])
    for i in range(3, len(A)):
        while pseudo_angle(stack_sub_top(S), stack_top(S), A[i]) <= 0:
            unstack(S)
        stack(S, A[i])
    return S


def image_hull(image, step=5, null_val=0., inside=True):
    """
    Compute the convex hull of a 2D image and return the 4 relevant coordinates
    of the maximum included rectangle (ie. crop image to maximum rectangle).
    ----------
    Inputs:
    image : numpy.ndarray
        2D array of floats corresponding to an image in intensity that need to
        be cropped down.
    step : int, optional
        For images with straight edges, not all lines and columns need to be
        browsed in order to have a good convex hull. Step value determine
        how many row/columns can be jumped. With step=2 every other line will
        be browsed.
        Defaults to 5.
    null_val : float, optional
        Pixel value determining the threshold for what is considered 'outside'
        the image. All border pixels below this value will be taken out.
        Defaults to 0.
    inside : boolean, optional
        If True, the cropped image will be the maximum rectangle included
        inside the image. If False, the cropped image will be the minimum
        rectangle in which the whole image is included.
        Defaults to True.
    ----------
    Returns:
    vertex : numpy.ndarray
        Array containing the vertex of the maximum included rectangle.
    """
    H = []
    shape = np.array(image.shape)
    row, col = np.indices(shape)
    for i in range(0,shape[0],step):
        r = row[i,:][image[i,:]>null_val]
        c = col[i,:][image[i,:]>null_val]
        if len(r)>1 and len(c)>1:
            H.append((r[0],c[0]))
            H.append((r[-1],c[-1]))
    for j in range(0,shape[1],step):
        r = row[:,j][image[:,j]>null_val]
        c = col[:,j][image[:,j]>null_val]
        if len(r)>1 and len(c)>1:
            if not((r[0],c[0]) in H):
                H.append((r[0],c[0]))
            if not((r[-1],c[-1]) in H):
                H.append((r[-1],c[-1]))
    S = np.array(convex_hull(H))

    x_min, y_min = S[:,0]<S[:,0].mean(), S[:,1]<S[:,1].mean()
    x_max, y_max = S[:,0]>S[:,0].mean(), S[:,1]>S[:,1].mean()
    # Get the 4 extrema
    S0 = S[x_min*y_min][np.abs(0-S[x_min*y_min].sum(axis=1)).min() == np.abs(0-S[x_min*y_min].sum(axis=1))][0]
    S1 = S[x_min*y_max][np.abs(shape[1]-S[x_min*y_max].sum(axis=1)).min() == np.abs(shape[1]-S[x_min*y_max].sum(axis=1))][0]
    S2 = S[x_max*y_min][np.abs(shape[0]-S[x_max*y_min].sum(axis=1)).min() == np.abs(shape[0]-S[x_max*y_min].sum(axis=1))][0]
    S3 = S[x_max*y_max][np.abs(shape.sum()-S[x_max*y_max].sum(axis=1)).min() == np.abs(shape.sum()-S[x_max*y_max].sum(axis=1))][0]
    # Get the vertex of the biggest included rectangle
    if inside:
        f0 = np.max([S0[0],S1[0]])
        f1 = np.min([S2[0],S3[0]])
        f2 = np.max([S0[1],S2[1]])
        f3 = np.min([S1[1],S3[1]])
    else:
        f0 = np.min([S0[0],S1[0]])
        f1 = np.max([S2[0],S3[0]])
        f2 = np.min([S0[1],S2[1]])
        f3 = np.max([S1[1],S3[1]])

    return np.array([f0, f1, f2, f3]).astype(int)
