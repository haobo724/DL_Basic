import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, BCELoss

# -------------------实例化所有Loss---------------------
CE_loss = CrossEntropyLoss(reduction='mean')
BCE_Loss = BCELoss(reduction='mean')
BCE_LOG_LOSS = BCEWithLogitsLoss(reduction='mean')

# -------------------设置Input和target---------------------

# target.shape=(B,W,H) Batch Width,Height
# 如果已经是one-hot或者smooth-label，即input.shape=target.shape的可以直接进Loss
target = torch.rand((4, 10, 12)).random_(0, 2).long()
target_random = torch.empty((4, 1, 10, 12)).fill_(1).long()
# 注意二分类的时候input的channel数是2而不是1，但也可以设置为1，见实验2
input = torch.ones((4, 2, 10, 12))
input_random = torch.zeros((4, 1, 10, 12))
input_random[:, 0, :, :] = 1
# temp = torch.log(input_random)
# print(torch.sum(temp))
target_onehot = torch.moveaxis(torch.nn.functional.one_hot(target).float(), -1, 1)
print(target_onehot.size())  # torch.Size([1, 2, 10, 12])
# -------------------开始loss计算-------------------------
# ---------实验一：二分类，input的C=2，分别使用：
#       1.用CE公式手动计算
#       2.BCE + softmax
#       3.CE
#       4.CE + target手动做onehot
#       5.BCELOG

#       先说结论，1,2,3,4 四个结果相等,另外要注意CE的输入格式要不然两个shape相同，要不然target无Channel，CE会根据input的shape自动做onehot
#       另外根据实验一中第五个小项和下面实验二中1-3小项结果的对比可知对于二分类在pytorch里BCE和CE公式等价因为label只有0,1,数学上一眼也看得出来

# 根据公式手动计算loss
input_softmax = torch.softmax(input, dim=1)
input_log = torch.log(input_softmax)
result_manuel = -torch.sum(target_onehot * input_log) / (target.shape[0] * target.shape[1] * target.shape[2])

# 使用pytorch集成库
BCE_Loss_result_butsoftmax = BCE_Loss.forward(input_softmax,
                                              target_onehot.float())  # input:(4, 2, 10, 12) target_onehot:(4, 2, 10, 12)

CE_loss_result = CE_loss.forward(input, target)  # input:(4, 2, 10, 12) target:(4, 10, 12)
CE_loss_result_target_onehot = CE_loss.forward(input, target_onehot)

print(result_manuel, CE_loss_result, CE_loss_result_target_onehot, BCE_Loss_result_butsoftmax)
# tensor(0.6931) tensor(0.6931) tensor(0.6931) tensor(0.6931)：证明CEloss的target会自动做onehot

BCE_LOG_LOSS_result_normal = BCE_LOG_LOSS.forward(input, target_onehot.float())
print(BCE_LOG_LOSS_result_normal)
# 且在二分类时使用softmax激活的BCE和CE结果相同, 如果使用sigmod激活，结果自然不同。
# BCE不会自动做onehot，即BCE的input和target必须要同shape, 另外用BCE进行多分类（C>2）时，因为公式不同，所以会和CE算出不一样结果

# ---------实验二：二分类，input的C=1，分别使用：
#       1.用BCE公式手动计算 用sigmoid
#       2.BCE + 手动sigmoid
#       3.BCElog
#       4.CE (target手动做onehot)
#       5.BCE

# 先说结论：1,2,3和预期相符的相同,即BCE_LOG_LOSS = BCE_Loss+ log(input)，此外需要注意，因为input和target需要同shape，所以target要unsqueeze一下.
#         5 因为没有sigmod使输入scale到0,1之间所以结果较大
#         第四个比较有意思, 可以说单通道的input没法正常使用pytorch 中 CE loss

# 根据公式手动计算loss
input_onechannel = torch.ones((4, 1, 10, 12))
input_sigmoid = torch.sigmoid(input_onechannel)
input_log2 = torch.log(input_sigmoid)
input_log3 = torch.log(1 - input_sigmoid)
result_manuel2 = -(torch.sum(target.unsqueeze(1) * input_log2 + (1 - target.unsqueeze(1)) * input_log3)) / (
        target.shape[0] * target.shape[1] * target.shape[2])

# 使用pytorch集成库
BCE_Loss_result_butsigmod = BCE_Loss.forward(torch.sigmoid(input_onechannel), target.unsqueeze(1).float())
BCE_LOG_LOSS_result = BCE_LOG_LOSS.forward(input_onechannel, target.unsqueeze(1).float())
print(result_manuel2, BCE_Loss_result_butsigmod, BCE_LOG_LOSS_result)

# 使用CE强行做单通道二分类
CE_loss_result_with_single_input_channel = CE_loss.forward(input_onechannel.squeeze(1), target.float())
print(CE_loss_result_with_single_input_channel)  # tensor(11.5609) ,错误示范，这样意思是C=10
CE_loss_result_with_single_input_channel2 = CE_loss.forward(input_onechannel, target)
print(
    CE_loss_result_with_single_input_channel2)  # tensor(11.5609) ,错误示范，会报错 IndexError: Target 1 is out of bounds.因为没法onehot

BCE_Loss_result = BCE_Loss.forward(input_onechannel, target.unsqueeze(1).float())
print(BCE_Loss_result)
