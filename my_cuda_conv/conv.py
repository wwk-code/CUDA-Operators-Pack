import numpy as np
from scipy.signal import convolve2d

# 输入数据
in_data = np.array([
    [1, 2, 3, 0, 1],
    [0, 1, 2, 3, 1],
    [1, 2, 0, 1, 2],
    [2, 0, 1, 1, 0],
    [1, 1, 2, 0, 1]
], dtype=np.float32)

# 卷积核
kernel = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
], dtype=np.float32)

# 执行卷积
# 需要将卷积核翻转
kernel_flipped = np.flipud(np.fliplr(kernel))
out_data = convolve2d(in_data, kernel_flipped, mode='valid')

# 打印输出
print("Output:")
print(out_data)




