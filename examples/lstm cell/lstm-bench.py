import numpy as np
import lstm

c = lstm.lstm()

# initializes the weights and the inputs randomly
input_size = 5
hidden_size = 10
batch_size = 20

W_bi = np.random.random((hidden_size, input_size)).astype(np.float32)

U_bi = np.random.random((hidden_size, hidden_size)).astype(np.float32)
b_bi = np.random.random((hidden_size)).astype(np.float32)
W_ig = np.random.random((hidden_size, input_size)).astype(np.float32)
U_ig = np.random.random((hidden_size, hidden_size)).astype(np.float32)
b_ig = np.random.random((hidden_size)).astype(np.float32)
W_fg = np.random.random((hidden_size, input_size)).astype(np.float32)
U_fg = np.random.random((hidden_size, hidden_size)).astype(np.float32)
b_fg = np.random.random((hidden_size)).astype(np.float32)
W_og = np.random.random((hidden_size, input_size)).astype(np.float32)
U_og = np.random.random((hidden_size, hidden_size)).astype(np.float32)
b_og = np.random.random((hidden_size)).astype(np.float32)
input = np.random.random((input_size,batch_size)).astype(np.float32)
prev_output = np.random.random((hidden_size,batch_size)).astype(np.float32)
prev_cell = np.random.random((hidden_size, batch_size)).astype(np.float32)

output = c.test(W_bi,input,b_bi)

#output = c.main(
#        W_bi,U_bi,b_bi,
#	    W_ig,U_ig,b_ig,
#	    W_fg,U_fg,b_fg,
#	    W_og,U_og,b_og,
#	    input,
#	    prev_output,
#	    prev_cell)
