import numpy as np
import pandas as pd
import copy

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

def find_neighbor(V):
	global matrix
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


def prepare_matrix(matrix_origin, answer):
	matrix = matrix_origin.copy()
	SWP = []
	for an in answer:
		for i in an:
			SWP.append(i)
	for row in range(len(matrix_origin)):
		for column in range(len(matrix_origin)):
			matrix[row][column] = matrix_origin[ SWP[row] ][ SWP[column] ]
	return matrix

def change_matrix(matrix, n, m):
	matrix[:, [n, m]] = matrix[:, [m, n]]
	matrix[[n, m], :] = matrix[[m, n], :]
	return matrix

def get_pos_by_group(group_global, value):
	pos = 0
	for group in group_global:
		if value not in group:
			pos += len(group)
		else:
			pos += np.argwhere(group == value)#group.index(value)
			return pos

def change_matrix_by_groups(matrix):
	global group_global
	matr = matrix.copy()
	SWP = []
	for group in group_global:
		for g in group:
			SWP.append(g)
	for row in range(len(matrix)):
		for column in range(len(matrix)):
			matr[row][column] = matrix[ SWP[row] ][ SWP[column] ]
	return matr

def get_col_by_j(j, groups):
	m = 0
	k_len = 0
	while k_len <= j:
		m += 1
		k_len += len(groups[m])
	k_len -= len(groups[m])
	return groups[m][j - k_len]

def get_row_by_i(i, groups):
	return groups[0][i]

def count_Q(answer, matrix):
	a, b, Q = (0, 0, 0)
	for i in range(len(answer)):
		a = b
		b += len(answer[i])
		Q += sum(matrix[a:b, b:30].sum(axis=1))
	return Q

def find_row_group(groups):
	return groups[0]

def find_column_group(groups, column):
	for r in groups:
		if column in r:
			return r

def S(n, group, matrix_origin):
	count = 0
	for i in group:
		count += matrix_origin[n][i]
	return count

def len_first_group(groups):
	return len(groups[0])

def len_rest_groups(groups):
	my_sum = 0
	for g in groups[1:]:
		my_sum += len(g)
	return my_sum

def count_delta_R(row, column, row_group, column_group, matrix_origin):
	return S(row, column_group, matrix_origin) - \
		   S(row, row_group, matrix_origin) + \
		   S(column, row_group, matrix_origin) - \
		   S(column, column_group, matrix_origin) - \
		   2 * matrix_origin[row][column]

def create_R_matrix(w, h):
	return [[0 for x in range(w)] for y in range(h)]

def fill_R_matrix(groups, matrix_origin, R_matrix, w, h):
	for i in range(h):
		for j in range(w):
			row = get_row_by_i(i, groups)
			col = get_col_by_j(j, groups)
			row_group = find_row_group(groups)
			column_group = find_column_group(groups, col)
			R_matrix[i][j] = count_delta_R(row, col, row_group, column_group, matrix_origin)
	return R_matrix

def find_positive_elem(R_matrix, w, h, groups):
	masR = []
	masInd = []
	for i in range(h):
		for j in range(w):
			if R_matrix[i][j] > 0:
				masR.append(R_matrix[i][j])
				masInd.append((i, j))
	if len(masR) == 0:
		return (-1, -1)
	r_max = np.argwhere(masR == max(masR))
	ind_max = masInd[r_max[0][0]]
	ind_max_ret = [0, 0]
	ind_max_ret[0] = get_row_by_i(ind_max[0], groups)
	ind_max_ret[1] = get_col_by_j(ind_max[1], groups)
	return ind_max_ret

def change_groups(n, m, groups):
	for i in range(len(groups)):
		for j in range(len(groups[i])):
			if groups[i][j] == n:
				groups[i][j] = m
				continue
			if groups[i][j] == m:
				groups[i][j] = n
				continue
	return groups 

def delete_optimized_group(groups):
	del groups[0]
	return groups

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
		min_ind = np.argmin(row_sum)
		V[num_group] = fill_V(V, min_ind, num_group)
		if len(V[num_group]) > N[num_group]:
			V[num_group] = cut_V_to_N(V[num_group], N[num_group])
		elif len(V[num_group]) < N[num_group]:
			V[num_group] = find_neighbors_to_V(V[num_group], N[num_group])
		answer[num_group] = indx[V[num_group]]
		complete_group_fill(V[num_group])
		num_group += 1

	for an in answer:
		print(an)

	group_global = copy.deepcopy(answer)
	groups = copy.deepcopy(answer)
	matrix = prepare_matrix(matrix_origin, answer)
	iter_num = 0
	while len(groups) > 1:
		iter_num += 1
		Q = count_Q(answer, matrix)
		print(Q)
		input()
		h = len_first_group(groups)
		w = len_rest_groups(groups)
		R_matrix = create_R_matrix(w, h)
		R_matrix = fill_R_matrix(groups, matrix_origin, R_matrix, w, h)
		row, col = find_positive_elem(R_matrix, w, h, groups)
		if row < 0 and col < 0:
			groups = delete_optimized_group(groups)
		else:
			row_m = get_pos_by_group(group_global, row)
			col_m = get_pos_by_group(group_global, col)
			matrix = change_matrix(matrix, row_m, col_m)
			groups = change_groups(row, col, groups)
			group_global = change_groups(row, col, group_global)
	print("Q is totally")
	print(count_Q(answer, matrix))
	print("iter_num ", iter_num)
	print("#################")
	input()

if __name__ == "__main__":
	m_file = open("combination.txt")
	A = []
	for f in m_file:
		l = [int(w) for w in f if w not in ['[', ']', ' ', ',', '\n']]
		l.sort()
		if l not in A:
			A.append(copy.copy(l))
			print("N == ", A[-1])
			main(A[-1])
