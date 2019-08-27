#!/usr/bin/env python3

import numpy as np
from numpy.linalg import lstsq
import sys
from math import sqrt
from itertools import product

A = np.array([
	[1, 0, 0, 0, 0, 0, 0],
	[1, 0, 0, 0, 0, 0, 0],
	[1, 0, 0, 0, 0, 0, 0],
	[1, 0, 0, 0, 0, 0, 0],
	[1, 1, 0, 0, 0, 0, 0],
	[1, 0, 1, 1, 0, 0, 0],
	[1, 0, 1, 1, 0, 0, 0],
	[1, 0, 1,-1, 0, 0, 0],
	[1, 0, 1,-1, 0, 0, 0],
	[0, 0, 0, 0, 1, 0, 0],
	[0, 0, 0, 0, 1, 0, 0],
	[0, 0, 0, 0, 1, 0, 0],
	[0, 0, 0, 0, 1, 0, 0],
	[0, 0, 0, 0, 0, 1, 0],
	[0, 0, 0, 0, 0, 1, 0],
	[0, 0, 0, 0, 0, 1, 0],
	[0, 0, 0, 0, 0, 1, 0],
	[0, 0, 0, 0, 0, 0, 1],
	[0, 0, 0, 0, 0, 0, 1],
	[0, 0, 0, 0, 0, 0, 1],
	[0, 0, 0, 0, 0, 0, 1]])
# A = np.ones((9,1))

def b(B):
	# return np.array([B[i,j] for i,j in product(*(([0, 4, 8],)*2))])
	return np.array([B[0,0], B[0,4], B[4,8], B[8,8], B[0,8], B[8,0], B[4,4],
		B[4,0], B[8,4], B[5,1], B[5,3], B[7,1], B[7,3], B[1,5], B[1,7], B[3,5],
		B[3,7], B[2,2], B[2,6], B[6,2], B[6,6]])

def onsite(C):
	N = int(sqrt(C.shape[0] / 9))
	a = (N + 1) * 9 * np.arange(N)
	return [C[i:(i+9), j:(j+9)] for i, j in product(a, a)]

def bond(C):
	N = int(sqrt(C.shape[0] / 9))
	a = (N + 1) * 9 * np.arange(N)
	c = 9 * np.arange(N**2)
	b = [x for x in c if not x in a]
	return [C[i:(i+9), j:(j+9)] for i, j in product(b, b)]

def cross(C):
	N = int(sqrt(C.shape[0] / 9))
	a = (N + 1) * 9 * np.arange(N)
	c = 9 * np.arange(N**2)
	b = [x for x in c if not x in a]
	return [C[i:(i+9), j:(j+9)] for i, j in product(c, c) if not ((i in a and j in a) or (i in b and j in b))]

def all(C):
	N = int(sqrt(C.shape[0] / 9))
	c = 9 * np.arange(N**2)
	return [C[i:(i+9), j:(j+9)] for i, j in product(c, c)]


def fit(blocks):
	N = len(blocks)
	return lstsq(np.concatenate((A,) * N),
		np.concatenate([b(B) for B in blocks]))[0]

C = np.loadtxt(sys.argv[1])
def format(x):
	return ' '.join(['{:8.3f}'.format(y) for y in x])

print(format(fit(onsite(C))))
print(format(fit(cross(C))))
print(format(fit(bond(C))))
print(format(fit(cross(C))/fit(onsite(C))))
print(format(fit(bond(C))/fit(cross(C))))
# print(format(np.concatenate([fit(onsite(C)),
# 	fit(cross(C)),
# 	fit(bond(C)),
# 	fit(cross(C))/fit(onsite(C)),
# 	fit(bond(C))/fit(cross(C))])))
# print((fit(bond(C))/fit(cross(C)))[0])

# Qz = [fit([x])[1] for x in all(C)]
# Qz /= max(Qz)
# Qz = np.array(Qz).reshape((16, 16))
# for row in Qz:
# 	print(format(row))
