from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net
from utils import *
from loss import OriTripletLoss, TripletLoss_WRT, TripletLoss_ADP
from tensorboardX import SummaryWriter
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu',
                    help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1, type=float,
                    help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str,
                    help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true',
                    help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int, metavar='s',
                    help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int, metavar='imgw',
                    help='img width')
parser.add_argument('--img_h', default=288, type=int, metavar='imgh',
                    help='img height')
parser.add_argument('--batch-size', default=6, type=int, metavar='B',
                    help='training batch size')
parser.add_argument('--test-batch', default=48, type=int, metavar='tb',
                    help='testing batch size')
parser.add_argument('--method', default='adp', type=str, metavar='m',
                    help='method type: base or agw, adp')
parser.add_argument('--margin', default=0.3, type=float, metavar='margin',
                    help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int, metavar='t',
                    help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t',
                    help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str,
                    help='all or indoor')
parser.add_argument('--augc', default=1, type=int, metavar='aug',
                    help='use channel aug or not')
parser.add_argument('--rande', default=0.5, type=float, metavar='ra',
                    help='use random erasing or not and the probability')
parser.add_argument('--alpha', default=1, type=int, metavar='alpha',
                    help='magnification for the hard mining')
parser.add_argument('--gamma', default=1, type=int, metavar='gamma',
                    help='gamma for the hard mining')
parser.add_argument('--square', default=1, type=int, metavar='square',
                    help='gamma for the hard mining')
parser.add_argument('--fsa_method', default='FSA', type=str,
                    help='Mix_A or Ex_P or FSA or None')

# 从命令行解析参数
args = parser.parse_args()
# 设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# 设置随机种子
set_seed(args.seed)

# 设置数据集
dataset = args.dataset
if dataset == 'sysu':
    # data_path = '/home/tom/PycharmProjects/FSA/Sysumm01'
    data_path = 'Sysumm01'
    log_path = args.log_path + 'sysu_log/' + '0.8/3/'
    # 将测试模式设置为 [1, 2]，表示将红外图像转换为可见光图像进行测试
    test_mode = [1, 2]
elif dataset == 'regdb':
    data_path = '/home/tom/PycharmProjects/FSA/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    # 将测试模式设置为 [2, 1]，表示将可见光图像转换为红外图像进行测试
    test_mode = [2, 1]

# 设置模型保存路径
checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

# 设置文件名后缀为数据集名称
suffix = dataset
# 如果训练方法为 ADP
if args.method == 'adp':
    # 设置文件名后缀为 ADP
    suffix += '_{}_joint_co_nog_ch_nog_sq{}'.format(args.method, args.square)
else:
    # 设置文件名后缀为训练方法
    suffix += '_{}'.format(args.method)
# 如果使用通道增强
if args.augc == 1:
    # 设置文件名后缀为通道增强
    suffix += '_aug_G'
# 如果设置了随机擦除的参数
if args.rande > 0:
    # 设置文件名后缀为随机擦除
    suffix += '_erase_{}'.format(args.rande)

# 增加后缀
suffix += '_p{}_n{}_lr_{}_seed_{}_fsa_method_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed, args.fsa_method)
# 如果优化器不为 SGD
if not args.optim == 'sgd':
    # 增加optim后缀
    suffix = suffix + '_' + args.optim
# 如果数据集为 RegDB
if dataset == 'regdb':
    # 增加trial后缀
    suffix += '_trial_{}'.format(args.trial)

# 将控制台输出重定向到指定的日志文件中
sys.stdout = Logger(log_path + suffix + '_os.txt')
# 定义可视化日志的保存路径
vis_log_dir = args.vis_log_path + suffix + '/'
# 如果可视化日志的保存路径不存在，则创建该路径
if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)

# 创建一个SummaryWriter对象，用于保存可视化日志信息
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
# 将数据转换为标准正太分布，使模型更容易收敛(标准化)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# 数据转换
transform_train_list = [
    # 将图像转换成PILImage
    transforms.ToPILImage(),
    # 对图像周围进行10个像素的填充
    transforms.Pad(10),
    # 将图像随机裁剪成288*146大小
    transforms.RandomCrop((args.img_h, args.img_w)),
    # 将图像随机水平翻转
    transforms.RandomHorizontalFlip(),
    # 将图像转换成Tensor
    transforms.ToTensor(),
    # 将图像标准化
    normalize]
# 测试集数据转换
transform_test = transforms.Compose([
    # 将图像转换成PILImage
    transforms.ToPILImage(),
    # 将图像裁剪成288*144大小
    transforms.Resize((args.img_h, args.img_w)),
    # 将图像转换成Tensor
    transforms.ToTensor(),
    # 将图像标准化
    normalize])
# 随机擦除
if args.rande > 0:
    transform_train_list = transform_train_list + [ChannelRandomErasing(probability=args.rande)]
# 通道增强
if args.augc == 1:
    transform_train_list = transform_train_list + [ChannelAdapGray(probability=0.5)]
transform_train = transforms.Compose(transform_train_list)

end = time.time()
if dataset == 'sysu':
    # 训练集为sysu数据集
    trainset = SYSUData(data_path, transform=transform_train)
    # 生成每个人身份的idx
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # 查询的图片，标签，相机
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    # 图库里的候选图片，标签，相机
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # 训练集为regdb数据集
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # 生成每个人身份的idx
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # 查询的图片，标签，查询的是可见光图像
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    # 图库里的候选图片，标签，查询的是热红外图像
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

# 数据封装
gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
# 数据加载
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=0)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=0)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
# 构建模型
# 加载的模型是否为基本模型，是则设置no_local='off'和gm_pool='off'，否则设置为no_local='on'和gm_pool='on'
if args.method == 'base':
    # 创建神经网络模型net
    net = embed_net(n_class, no_local='off', gm_pool='off', arch=args.arch, fsa_method=args.fsa_method)
else:
    net = embed_net(n_class, no_local='on', gm_pool='on', arch=args.arch, fsa_method=args.fsa_method)
# net传送到设备上
net.to(device)
# 使用cudnn加速
cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# 定义损失函数(交叉熵损失函数)
criterion_id = nn.CrossEntropyLoss()
if args.method == 'agw':
    # 加权正则化三元组损失
    criterion_tri = TripletLoss_WRT()
    # loader_batch = args.batch_size * args.num_pos
    # criterion_tri= OriTripletLoss(batch_size=loader_batch, margin=args.margin)
elif args.method == 'adp':
    # 自适应距离惩罚三元组损失
    criterion_tri = TripletLoss_ADP(alpha=args.alpha, gamma=args.gamma, square=args.square)
else:
    loader_batch = args.batch_size * args.num_pos
    criterion_tri = OriTripletLoss(batch_size=loader_batch, margin=args.margin)
# 推送到GPU计算
criterion_id.to(device)
criterion_tri.to(device)


if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)


# 定义调整学习率
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr


# 定义训练函数
def train(epoch):  # 定义训练函数
    current_lr = adjust_learning_rate(optimizer, epoch)  #
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0
    # switch to train mode
    net.train()
    end = time.time()
    for batch_idx, (input10, input11, input20, label1, label2) in enumerate(trainloader):

        labels = torch.cat((label1, label1, label2), 0)
        labels = Variable(labels.cuda())

        input10 = Variable(input10.cuda())
        input11 = Variable(input11.cuda())
        input1 = Variable(torch.cat((input10, input11), 0).cuda())
        input20 = Variable(input20.cuda())

        data_time.update(time.time() - end)

        feat, out0, = net(input10, input11, input20)

        loss_id = criterion_id(out0, labels.long())

        loss_tri, batch_acc = criterion_tri(feat, labels)
        correct += (batch_acc / 2)
        _, predicted = out0.max(1)
        correct += (predicted.eq(labels).sum().item() / 2)

        # pdb.set_trace()
        loss = loss_id + loss_tri
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update P
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri.item(), 2 * input1.size(0))
        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{:.3f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'Accu: {:.2f}'.format(
                epoch, batch_idx, len(trainloader), current_lr,
                100. * correct / total, batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss))

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)


def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 2048))
    gall_feat_att = np.zeros((ngall, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, input, test_mode[0])
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 2048))
    query_feat_att = np.zeros((nquery, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, input, test_mode[1])
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))

    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)
        cmc_att, mAP_att, mINP_att = eval_regdb(-distmat_att, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    writer.add_scalar('rank1', cmc[0], epoch)
    writer.add_scalar('mAP', mAP, epoch)
    writer.add_scalar('mINP', mINP, epoch)
    writer.add_scalar('rank1_att', cmc_att[0], epoch)
    writer.add_scalar('mAP_att', mAP_att, epoch)
    writer.add_scalar('mINP_att', mINP_att, epoch)
    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att


# training
print('==> Start Training...')
print(start_epoch)
for epoch in range(start_epoch, 100 - start_epoch):
    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, trainset.train_thermal_label, color_pos, thermal_pos,
                              args.num_pos, args.batch_size, epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)
    print(trainset.cIndex)
    print(trainset.tIndex)
    loader_batch = args.batch_size * args.num_pos
    trainloader = data.DataLoader(trainset, batch_size=loader_batch, sampler=sampler, num_workers=0,
                                  drop_last=True)
    # training
    train(epoch)

    if epoch >= 0:  # and epoch % 2 == 0:
        print('Test Epoch: {}'.format(epoch))
        # testing
        cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = test(epoch)
        # save model
        if cmc_att[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc_att[0]
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'cmc': cmc_att,
                'mAP': mAP_att,
                'mINP': mINP_att,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')
        print(
            'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'
            .format(cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'
            .format(cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
        print('Best Epoch [{}]'.format(best_epoch))

