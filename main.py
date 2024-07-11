import torch
import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import train
from src.eval_metrics import eval_mosi,eval_iemocap,eval_mosei_senti


# 创建一个解析器对象来处理命令行参数
parser = argparse.ArgumentParser(description='MOSEI情感分析')

# 用于Jupyter环境中的辅助参数，通常用于避免在notebooks中使用argparse时出现错误
parser.add_argument('-f', default='', type=str)

# 固定参数，指定使用的模型，默认为'MulT'
parser.add_argument('--model', type=str, default='MulT', help='要使用的模型名称（如Transformer等）')

# 参数指定独占使用哪种模态
parser.add_argument('--vonly', action='store_true', help='仅使用视觉模态的交叉模态融合（默认为False）')
parser.add_argument('--aonly', action='store_true', help='仅使用音频模态的交叉模态融合（默认为False）')
parser.add_argument('--lonly', action='store_true', help='仅使用语言模态的交叉模态融合（默认为False）')

# 参数指定是否对数据集进行对齐
parser.add_argument('--aligned', action='store_true', help='是否考虑对齐实验（默认为False）')
# 指定数据集和路径
parser.add_argument('--dataset', type=str, default='iemocap', help='要使用的数据集（默认为mosei_senti）')
parser.add_argument('--data_path', type=str, default='data', help='存储数据集的路径')

# 模型各部分的dropout配置
parser.add_argument('--attn_dropout', type=float, default=0.1, help='注意力层的dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.0, help='音频注意力层的dropout')
parser.add_argument('--attn_dropout_v', type=float, default=0.0, help='视觉注意力层的dropout')
parser.add_argument('--relu_dropout', type=float, default=0.1, help='ReLU层的dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25, help='嵌入层的dropout')
parser.add_argument('--res_dropout', type=float, default=0.1, help='残差块的dropout')
parser.add_argument('--out_dropout', type=float, default=0.0, help='输出层的dropout')

# 架构规格
parser.add_argument('--nlevels', type=int, default=5, help='网络中的层数（默认为5）')
parser.add_argument('--num_heads', type=int, default=5, help='Transformer网络的头数（默认为5）')
parser.add_argument('--attn_mask', action='store_false', help='是否使用Transformer的注意力掩码（默认为真）')

# 训练超参数
parser.add_argument('--batch_size', type=int, default=24, metavar='N', help='批量大小（默认为24）')
parser.add_argument('--clip', type=float, default=0.8, help='梯度裁剪值（默认为0.8）')
parser.add_argument('--lr', type=float, default=1e-3, help='初始学习率（默认为1e-3）')
parser.add_argument('--optim', type=str, default='Adam', help='要使用的优化器（默认为Adam）')
parser.add_argument('--num_epochs', type=int, default=40, help='训练周期数（默认为40）')
parser.add_argument('--when', type=int, default=20, help='何时对学习率进行衰减（默认为20）')
parser.add_argument('--batch_chunk', type=int, default=1, help='每批次的分块数（默认为1）')

# 实验运行的逻辑
parser.add_argument('--log_interval', type=int, default=30, help='结果记录的频率（默认为30）')
parser.add_argument('--seed', type=int, default=1111, help='随机种子')
parser.add_argument('--no_cuda', action='store_true', help='不使用cuda')
parser.add_argument('--name', type=str, default='mult', help='试验的名称（默认为"mult"）')

# 解析参数
args = parser.parse_args()

# 设置随机种子以确保结果的可重现性
torch.manual_seed(args.seed)
dataset = str.lower(args.dataset.strip())
valid_partial_mode = args.lonly + args.vonly + args.aonly

# 确保只选择了一种模态，否则抛出错误
if valid_partial_mode == 0:
    args.lonly = args.vonly = args.aonly = True
elif valid_partial_mode != 1:
    raise ValueError("您只能选择{l/v/a}only中的一个。")

use_cuda = False

# 根据数据集确定输出维度
output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
    'iemocap': 8
}

# 根据数据集选择损失函数
criterion_dict = {
    'iemocap': 'CrossEntropyLoss'
}

# 设置默认张量类型
torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("警告：您有一个CUDA设备，因此最好不要运行--no_cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True

####################################################################
#
# 加载数据集（对齐或非对齐）
#
####################################################################

print("开始加载数据....")

train_data = get_data(args, dataset, 'train')
valid_data = get_data(args, dataset, 'valid')
test_data = get_data(args, dataset, 'test')

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

print('数据加载完毕....')
if not args.aligned:
    print("### 注意：您正在运行非对齐模式。")

####################################################################
#
# 超参数
#
####################################################################

hyp_params = args
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_data.get_dim()
hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_data.get_seq_len()
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = criterion_dict.get(dataset, 'L1Loss')
if __name__ == '__main__':
    test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)


