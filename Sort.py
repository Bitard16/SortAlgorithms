import timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import random
import GenerateSOme as gs

def selection(arrayToSort):
    a = arrayToSort.copy()
    for i in range(len(a)):
        idxMin = i
        for j in range(i+1, len(a)):
            if a[j] < a[idxMin]:
                idxMin = j
        tmp = a[idxMin]
        a[idxMin] = a[i]
        a[i] = tmp
    return a

def insertion(arrayToSort):
    a = arrayToSort.copy()
    for i in range(len(a)):
        v = a[i]
        j = i
        while (a[j-1] > v) and (j > 0):
            a[j] = a[j-1]
            j = j - 1
        a[j] = v
    return a

def bubble(arrayToSort):
    a = arrayToSort.copy()
    for i in range(len(a),0,-1):
        for j in range(1, i):
            if a[j-1] > a[j]:
                tmp = a[j-1]
                a[j-1] = a[j]
                a[j] = tmp
    return a

def py_sorted(arrayToSort):
    a = arrayToSort.copy()
    return sorted(a)


def Shell(arrayToSort):
    a = arrayToSort.copy()
    inc = len(a) // 2
    while inc:
        for i, el in enumerate(a):
            while i >= inc and a[i - inc] > el:
                a[i] = a[i - inc]
                i -= inc
            a[i] = el
        inc = 1 if inc == 2 else int(inc * 5.0 / 11)
    return a


def quicksort(arrayToSort):
    a = arrayToSort.copy()
    less = []
    equal = []
    greater = []

    if len(a) > 1:
        pivot = a[0]
        for x in a:
            if x < pivot:
                less.append(x)
            elif x == pivot:
                equal.append(x)
            elif x > pivot:
                greater.append(x)
        # Don't forget to return something!
        return quicksort(less) + equal + quicksort(greater)  # Just use the + operator to join lists
    # Note that you want equal ^^^^^ not pivot
    else:  # You need to hande the part at the end of the recursion - when you only have one element in your array, just return the array.
        return a



# The main function to sort an data array of given size
def heapSort(arrayToSort):
    a = arrayToSort.copy()
    def heapify(a, n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2

        # Compare left child and root
        if l < n and a[i] < a[l]:
            largest = l

        # Compare right child and root
        if r < n and a[largest] < a[r]:
            largest = r

        # Change root, if needed
        if largest != i:
            a[i], a[largest] = a[largest], a[i]

            # Heapify the root
            heapify(a, n, largest)
    n = len(a)
    # Build a maxheap
    for i in range(n, -1, -1):
        heapify(a, n, i)

    # One by one extract elements
    for i in range(n - 1, 0, -1):
        a[i], a[0] = a[0], a[i]
        heapify(a, i, 0)
    return a



res = pd.DataFrame(
    index=['selection', 'insertion', 'bubble', 'py_sorted','Shell','quicksort','heapSort'],
    columns=np.logspace(2, 3, 6).astype(int),
    dtype=int
)

for j in res.columns:
    a = np.random.choice(j, j, replace=True)
    for i in res.index:
        stmt = '{}(a)'.format(i)
        setp = 'from __main__ import a, {}'.format(i)
        print('processing [{}]\tarray size: {}'.format(i,j), end='')
        res.at[i, j] = timeit.timeit(stmt, setp, number=50)
        print('\t\ttiming:\t{}'.format(res.at[i, j]))
print(res)

plt.figure()
ax = res.T.plot(loglog=True, style='-o', figsize=(10,8))
ax.set_xlabel('array size')
ax.set_ylabel('time (sec)')
plt.savefig('result.png')