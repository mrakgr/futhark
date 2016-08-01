-- Standard matrix matrix multiplication
-- R - regular matrix dimensions
-- F - Futhark matrix dimensions
-- F(o,m) = R(m,o)
-- F(n,o) = R(o,n)
-- R(m,o) * R(o,n) = R(m,n) = F(n,m)
fun [[f32,n],m] matmult([[f32,o],m] a, [[f32,n],o] b) =
	let b = transpose(b) in
	map(fn [f32,n] ([f32,o] a) =>
			map(fn f32 ([f32,o] b) =>
				reduce(+,0f32,zipWith(*,a,b))
			,b)
		,a)

fun [[f32,n],m] matmultImp([[f32,o],m] a, [[f32,n],o] b) =
    let res = replicate(m, replicate(n,0f32)) in
    loop (res) = for i < m do
        loop (res) = for j < n do
            let partsum =
                let res = 0f32 in
                loop (res) = for k < o do
                    let res = res + a[i,k] * b[k,j]
                    in  res
                in res
            in let res[i,j] = partsum in res
        in res
    in res

-- Adds two matrices
fun *[[f32]] add([[f32]] x, [[f32]] y) =
	zipWith(fn [f32] ([f32] x, [f32] y) =>
				zipWith(fn f32 (f32 x, f32 y) => x + y
				,x,y)
			,x,y)

-- Elementwise multiplies two matrices
fun *[[f32]] hadmult([[f32]] x, [[f32]] y) =
	zipWith(fn [f32] ([f32] x, [f32] y) =>
				zipWith(fn f32 (f32 x, f32 y) => x * y
				,x,y)
			,x,y)

-- Broadcast addition
fun [[f32,m],n] broadcast_add([[f32,m],n] a, [f32,n] b) =
	let b = transpose(replicate(m,b)) in
	add(a,b)

-- Ax + b
-- The standard feedforward neural net linear layer
fun [[f32,n],m] linear_layer([[f32,o],m] A, [[f32,n],o] x, [f32,m] b) =
    broadcast_add(matmult(A, x), b)

-- Ax + Bh + b
-- The recurrent neural net linear layer
fun [[f32,n],m] linear_layer_2([[f32,o],m] A, [[f32,n],o] x, [[f32,m],m] B, [[f32,n],m] h, [f32,m] b) =
    broadcast_add(add(matmult(A, x), matmult(B,h)), b)  

fun f32 exp(f32 a) =
	let e = 2.71828f32 in
	e**a -- Futhark needs more math functions. e ~= 2.71828.

fun f32 sigmoid(f32 a) = 1.0f32 / (1.0f32 + exp(-a))

-- The sigmoid activation function
-- All output elements should be in the [0,1] range
fun [[f32]] sigmoid_activation([[f32]] a) =
	map(fn [f32] ([f32] a) =>
			map (sigmoid,a)
		,a)

-- Feedforward Neural net layer with a sigmoid activation
fun [[f32,n],m] sigmoid_layer([[f32,o],m] A, [[f32,n],o] x, [f32,m] b) =
	sigmoid_activation(linear_layer(A,x,b))

-- Recurrent neural net layer with a sigmoid activation
fun [[f32,n],m] sigmoid_recurrent_layer([[f32,o],m] A, [[f32,n],o] x, [[f32,m],m] B, [[f32,n],m] h, [f32,m] b) =
	sigmoid_activation(linear_layer_2(A,x,B,h,b))

-- The LSTM cell/layer forward pass. Outputs a (output, cell) tuple.
-- This is the standard LSTM formula, without peepholes.
-- At the current time of writting 5/9/2016, LSTM is the most widely used
-- recurrent network architecture and has been for the past decade or so.
fun ([[f32,n],m], [[f32,n],m]) 
    lstm_cell_forward([[f32,o],m] W_bi, [[f32,m],m] U_bi, [f32,m] b_bi, 
					  [[f32,o],m] W_ig, [[f32,m],m] U_ig, [f32,m] b_ig,
					  [[f32,o],m] W_fg, [[f32,m],m] U_fg, [f32,m] b_fg,
					  [[f32,o],m] W_og, [[f32,m],m] U_og, [f32,m] b_og,
					  [[f32,n],o] input, 
					  [[f32,n],m] prev_output,
					  [[f32,n],m] prev_cell ) = 
	let block_input = sigmoid_recurrent_layer(W_bi,input,U_bi,prev_output,b_bi) -- In a real LSTM, this activation function is generally tanh, but can be sigmoid.
	let input_gate = sigmoid_recurrent_layer(W_ig,input,U_ig,prev_output,b_ig) -- The activations for the gates are always sigmoid
	let forget_gate = sigmoid_recurrent_layer(W_fg,input,U_fg,prev_output,b_fg)
	let output_gate = sigmoid_recurrent_layer(W_og,input,U_og,prev_output,b_og)
	let cell_state = add(hadmult(block_input,input_gate),hadmult(prev_cell,forget_gate)) -- As the backpropagation step is linear, the flow of gradients though the cell allow to go far further than in vanilla RNNs. As a consequence, LSTMs can learn sequences with far longer time lags.
	let output = hadmult(output_gate, sigmoid_activation(cell_state)) -- Here the activation can also be other than sigmoid or even left out entirely.
	in (output, cell_state)


fun ([[f32,n],m], [[f32,n],m]) 
                 main([[f32,o],m] W_bi, [[f32,m],m] U_bi, [f32,m] b_bi, 
					  [[f32,o],m] W_ig, [[f32,m],m] U_ig, [f32,m] b_ig,
					  [[f32,o],m] W_fg, [[f32,m],m] U_fg, [f32,m] b_fg,
					  [[f32,o],m] W_og, [[f32,m],m] U_og, [f32,m] b_og,
					  [[f32,n],o] input, 
					  [[f32,n],m] prev_output,
					  [[f32,n],m] prev_cell ) = 
	lstm_cell_forward
		(W_bi,U_bi,b_bi,
		 W_ig,U_ig,b_ig,
		 W_fg,U_fg,b_fg,
		 W_og,U_og,b_og,
		 input,
		 prev_output,
		 prev_cell)