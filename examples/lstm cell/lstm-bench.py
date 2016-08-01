import numpy as np
import lstm
import time

c = lstm.lstm()

# initializes the weights and the inputs randomly
input_size = 784
hidden_size = 256
batch_size = 64

W_bi = np.random.random((hidden_size, input_size)).astype(np.float32)
U_bi = np.random.random((hidden_size, hidden_size)).astype(np.float32)
b_bi = np.random.random((hidden_size,1)).astype(np.float32) # Remove the 1 from the 4 biases to trigger the code generation bug.
W_ig = np.random.random((hidden_size, input_size)).astype(np.float32)
U_ig = np.random.random((hidden_size, hidden_size)).astype(np.float32)
b_ig = np.random.random((hidden_size,1)).astype(np.float32)
W_fg = np.random.random((hidden_size, input_size)).astype(np.float32)
U_fg = np.random.random((hidden_size, hidden_size)).astype(np.float32)
b_fg = np.random.random((hidden_size,1)).astype(np.float32)
W_og = np.random.random((hidden_size, input_size)).astype(np.float32)
U_og = np.random.random((hidden_size, hidden_size)).astype(np.float32)
b_og = np.random.random((hidden_size,1)).astype(np.float32)
input = np.random.random((input_size,batch_size)).astype(np.float32)
prev_output = np.random.random((hidden_size,batch_size)).astype(np.float32)
prev_cell = np.random.random((hidden_size, batch_size)).astype(np.float32)

assert (np.dot(W_bi,input).shape == prev_output.shape)
assert (np.dot(U_bi,prev_output).shape == prev_output.shape)


output = c.main(
        W_bi,U_bi,b_bi,
	    W_ig,U_ig,b_ig,
	    W_fg,U_fg,b_fg,
	    W_og,U_og,b_og,
	    input,
	    prev_output,
	    prev_cell)
now = time.time()
for i in xrange(100):
    output = c.main(
            W_bi,U_bi,b_bi,
	        W_ig,U_ig,b_ig,
	        W_fg,U_fg,b_fg,
	        W_og,U_og,b_og,
	        input,
	        prev_output,
	        prev_cell)
end = time.time()

print output
print "Time elapsed: %f" % (end-now)