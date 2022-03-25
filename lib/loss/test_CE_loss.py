import torch
import torch.nn.functional as F
import torch.nn as nn
from lib.loss.loss_utils import encode_one_hot_label
torch.manual_seed(2019)
# shape: (2,3)
"""
tensor([[-0.1187,  0.2110,  0.7463],
        [-0.6136, -0.1186,  1.5565]])
"""
output = torch.randn(2, 3)  # 网络输出
print('pred',output)
# shape: (2)
"""
tensor([2, 0])
"""
target = torch.ones(2, dtype=torch.long).random_(3)  # 真实标签
print('ground truth',target)

print(encode_one_hot_label(pred=output,target=target))

# 实例化类
criterion = nn.CrossEntropyLoss()
#
loss = criterion(output, target)
print('loss:{}'.format(loss))

"""
output:
tensor([[-0.1187,  0.2110,  0.7463],
        [-0.6136, -0.1186,  1.5565]])
target:
tensor([2, 0])

log_softmax:
tensor([[-1.5614, -1.2317, -0.6964],
        [-2.4335, -1.9386, -0.2635]])

nll_loss:1.564985990524292
cross_entropy_loss:1.564985990524292
loss:1.564985990524292
"""
