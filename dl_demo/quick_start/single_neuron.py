import tensorlayerx as tlx
#tensorlayerx is a versatile ai training framework, including TensorFlow, PyTorch and so on

#Single neuro
x = tlx.constant([[1, 2, 3]])
w = tlx.constant([[-0.5], [0.2], [0.1]])

print("X:\n", x, "\nShape:", x.shape)
print("W:\n", w, "\nShape:", w.shape)

#Bias
b1 = tlx.constant(0.5)

z1 = tlx.matmul(x , w) + b1

print("Z:\n" ,z1, "\nShape:", z1.shape)

#cant run because tlx using tensorflow as backend
#trying to write by pytorch