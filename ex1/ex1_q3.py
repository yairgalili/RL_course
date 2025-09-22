# ’Bob’, ’Ok’, ’B’, ’Book’, ’Booook’
# p('Bob')=0.25*0.2*0.325
# p('Ok')=0
# p('B')=0.325
#p('Book')=0.25*0.2*0.2*0.2
#p(’Booook’)=0.25*0.2**3*0.2*0.2

# state space : b, k, o
# the action space: b, k, o
# the multiplicative cost function:
# prod(r_m, m=1, m=K-1)
# r_m=P(s_(m-1), s_m)

# complexity<o(3^K)

# sum(log(r_m), m=1, m=K-1)

# additive - less computational complexity, decrease slowly
# multiplicative - accurate, meaningful cost
import numpy as np

# Similar to Viterbi's algorithm:
def probable_word(A_orig, states, p0, k):
 	""""
 	Inputs:
 		1) transition matrix, Aij=P(sj|si), where the last row/ column is suitable foe the end of word.
 		2) k is the length of the word to be checked.
 		probable_words[:m, k] contains the m+1(]()) length most probable word, which ends with the k-th character
 	"""
 	
 	num_chars_language = A_orig.shape[0]-1
 	A = A_orig[:num_chars_language, :num_chars_language]
 	probable_words = -1 * np.ones((k-1, num_chars_language), dtype=int)
 	p = np.reshape(p0, (num_chars_language, 1))
 	
 	for ind_char in range(k-1):
 		Ap = A*p
 		probable_words[ind_char] = np.argmax(Ap, axis=0).astype(int)
 		print('Ap', Ap, sep='\n')
 		print('probable_words', probable_words, sep='\n')
 		p = np.reshape(Ap[probable_words[ind_char].astype(int), list(range(num_chars_language)) ], (num_chars_language, 1))
 		print('p', p)
 	
 	p_final = A_orig[:-1, -1]*np.squeeze(p)
 	ind_last = np.argmax(p_final)
 	str_opt = [states[probable_words[ind_char, ind_last]] for ind_char in range(k-1)]
 	str_opt.append(states[ind_last])
 	print(str_opt, p_final[ind_last])
 		
 			
if __name__=='__main__':
 	states= 'bko'
 	A = np.array([[0.1, 0.325, 0.25, 0.325], [0.4, 0, 0.4, 0.2], [0.2, 0.2, 0.2, 0.4], [1, 0, 0, 0]])
 	p0 = np.array([1, 0, 0])
 	k = 5
 	probable_word(A, states, p0, k)
 	