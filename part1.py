import numpy as np
import pandas as pd
import copy
"""
N = [4, 4, 7, 7, 7]
V = []
answer = []
for _ in range(len(N)):
	V.append([])
	answer.append([])
"""
df = pd.read_csv('data.csv', header=None)
matrix_origin = df.as_matrix()


def fill_V(V, min_ind, num_group):
	global indx
	global matrix
	V[num_group].append(min_ind)
	Verts = np.argwhere(matrix[min_ind] > 0)
	for v in Verts:
		V[num_group].append(v[0])
	V[num_group].sort()
	return V[num_group]

def exclude_index_from_V(V):
	global matrix
	delta = [0.0 for _ in range(len(V))]
	sum_row = [0.0 for _ in range(len(V))]
	sum_sik = [0.0 for _ in range(len(V))]
	for i in range(len(V)):
		sum_row[i] = sum(matrix[V[i]])
		for v in V:
			sum_sik[i] += matrix[V[i]][v]
		delta[i] = sum_row[i] - sum_sik[i]
	index = np.argmax(delta)
	#print("index == ", index)
	return index

def cut_V_to_N(V, N):
	while len(V) > N:
		index = exclude_index_from_V(V)
		del V[index]
	return V

def find_neighbors_to_V(V, N):
	global matrix
	while len(V) < N:
		V_k = find_neighbor(V)
		if V_k is not None:
			V = copy.copy(V_k)
	return V

def invert_from_indx(n):
	global indx
	#print("FIND FROM INDX")
	#print(indx, n)
	return np.argwhere(indx == n)

def find_neighbor(V):
	global matrix
	#print("FIND")
	#print(matrix)
	for row in V:
		for j in range(len(matrix)):
			if matrix[row][j] == 0:
				continue
			if j in V:
				continue
			V.append(j)
			return V
	return V

def complete_group_fill(V):
	global indx
	global matrix
	global matrix
	indx = np.delete(indx, V, axis=0)
	matrix = np.delete(matrix, V, axis=0)
	matrix = np.delete(matrix, V, axis=1)

def main(N):
	global matrix
	global indx

	matrix = matrix_origin.copy()
	indx = np.array([i for i in range(len(matrix_origin))])
	V, answer = [], []
	for _ in range(len(N)):
		V.append([])
		answer.append([])

	num_group = 0
	while num_group < len(N):
		row_sum = matrix.sum(axis=1)
		#print(matrix)
		#print(row_sum)
		#input()
		min_ind = np.argmin(row_sum)
		#print("min_ind == ", min_ind)
		V[num_group] = fill_V(V, min_ind, num_group)
		#print("V[num] ", V[num_group])
		#print("indx ", indx)
		#print("ind[V[num]] ", indx[V[num_group]])

		if len(V[num_group]) > N[num_group]:
			#print("V > N | need exclude")
			V[num_group] = cut_V_to_N(V[num_group], N[num_group])
			answer[num_group] = indx[V[num_group]]
			complete_group_fill(V[num_group])
			#print(V[num_group])
			#print(matrix)
			#print(answer)
			num_group += 1

		elif len(V[num_group]) == N[num_group]:
			#print("V == N")
			answer[num_group] = indx[V[num_group]]
			complete_group_fill(V[num_group])
			#print(V[num_group])
			#print(matrix)
			#print(answer)
			num_group += 1

		else:
			#print("V < N")
			#print(matrix)
			V[num_group] = find_neighbors_to_V(V[num_group], N[num_group])
			answer[num_group] = indx[V[num_group]]
			complete_group_fill(V[num_group])
			#print(V[num_group])
			#input()
			num_group += 1

	for an in answer:
		print(an)
	print("#################")


if __name__ == "__main__":
	m_file = open("combination.txt")
	A = []
	for f in m_file:
		l = [int(w) for w in f if w not in ['[', ']', ' ', ',', '\n']]
		#l.sort()
		if l not in A:
			A.append(copy.copy(l))
			print("N == ", A[-1])
			main(A[-1])