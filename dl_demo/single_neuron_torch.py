import os
os.environ["TL_BACKEND"] = "torch"
import tensorlayerx as tlx

x = xlt.constant([[1, 2, 3]])
w = tlx.constant([[-0.5], [0.2], [0.1]])
b1 = tlx.constant(0.5)

z1 = tlx.matmul(x, w) + b1
print(z1, z1.shape)

