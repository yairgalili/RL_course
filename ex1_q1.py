# longest common subsequence
# L(m,n)= longest common subsequence of x[:m+1] and y[:n+1]
# I(m, n)= last index of correspondence in y[:n+1] between x[:m+1] and y[:n+1]


# L(m, n) = L(m-1, n) + (x[m] in y[I(m-1,n):n+1])

def longest_common_subsequence(x_vec ,y_vec):
	length_detection = 0
	ind_last_detection = -1
	for i_x, x in enumerate(x_vec):
		for i_y in range(ind_last_detection + 1, len(y_vec)):
			if(x_vec[i_x] == y_vec[i_y]):
				ind_last_detection = i_y
				length_detection += 1
				break
	return length_detection


if __name__ == '__main__':
	x = 'AVBVAMCD'
	y = 'AZBQACLD'
	print(longest_common_subsequence(x, y))
				
			