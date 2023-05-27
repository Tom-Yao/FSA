# -*-coding:utf-8-*-
import torch
import numpy as np


# 提取振幅fft_amp和相位fft_pha
def extract_ampl_phase(fft_im):  # fft_im为输入的频谱图
    fft_amp = torch.sqrt(fft_im[:, :, :, :, 0] ** 2 + fft_im[:, :, :, :, 1] ** 2)  # 计算振幅
    fft_pha = torch.atan2(fft_im[:, :, :, :, 1], fft_im[:, :, :, :, 0])  # 计算相位
    return fft_amp, fft_pha  # 返回振幅和相位

# amp_x1为可见光图像的振幅，amp_x2为红外图像的振幅，L为低频成分的比例,lam为α
def low_freq_mutate(amp_x1, amp_x2, L=0.03, lam=0.5):
    # lam =  (0.5-alpha,0.5+alpha)     #lam为beta分布的参数，alpha为超参数
    # lam = torch.from_numpy(np.random.normal(0.5,alpha,size=amp_x1.size())).cuda()  #lam为正态分布的参数，alpha为超参数
    _, _, h, w = amp_x1.size()       # amp_x1和amp_x2的尺寸相同
    b = (np.floor(np.amin((h, w)) * L)).astype(int)  # b为低频成分的边长

    amp_x1_clone = amp_x1.clone()  # amp_x1_clone为amp_x1的深拷贝
    amp_x1_clone[:, :, 0:b, 0:b] = (1 - lam) * amp_x1[:, :, 0:b, 0:b] + lam * amp_x2[:, :, 0:b, 0:b]  # top left
    amp_x1_clone[:, :, 0:b, w - b:w] = (1 - lam) * amp_x1[:, :, 0:b, w - b:w] + lam * amp_x2[:, :, 0:b,
                                                                                      w - b:w]  # top right
    amp_x1_clone[:, :, h - b:h, 0:b] = (1 - lam) * amp_x1[:, :, h - b:h, 0:b] + lam * amp_x2[:, :, h - b:h,
                                                                                      0:b]  # bottom left
    amp_x1_clone[:, :, h - b:h, w - b:w] = (1 - lam) * amp_x1[:, :, h - b:h, w - b:w] + lam * amp_x2[:, :, h - b:h,
                                                                                              w - b:w]  # bottom right

    return amp_x1_clone  # 返回低频成分融合后的振幅


# PyTorch中的赋值运算符是浅拷贝！x4=x2.clone()而不是x4=x2，因为后者是浅拷贝，x4和x2共享内存空间，在对x4执行in-place运算后，x2的值也会随之改变
# clone()是深拷贝
# x_aug^(mix_a)=F^(-1) [[(1-α)A(x)+αA(x^' )] e^(-jP(x) ) ]
def mixup(x1, x2):  # x1为可见光图像，x2为红外图像
    fft_x1 = torch.fft.fft2(x1.clone(), dim=(-2, -1))  # 对输入进行二维离散傅里叶变换
    fft_x1 = torch.stack((fft_x1.real, fft_x1.imag), dim=-1)  # 连接实部和虚部
    amp_x1, pha_x1 = extract_ampl_phase(fft_x1.clone())  # 提取振幅和相位

    # 对另一张图片操作
    fft_x2 = torch.fft.fft2(x2.clone(), dim=(-2, -1))  # 对输入进行二维离散傅里叶变换
    fft_x2 = torch.stack((fft_x2.real, fft_x2.imag), dim=-1)  # 连接实部和虚部
    amp_x2, pha_x2 = extract_ampl_phase(fft_x2.clone())  # 提取振幅和相位

    # amp_x1_new = [(1-α)A(x)+αA(x^' )]
    amp_x1_new = low_freq_mutate(amp_x1=amp_x1.clone(), amp_x2=amp_x2.clone(), L=0.03, lam=0.5)  # 低频成分融合
    fft_clone = fft_x1.clone().cuda()  # 创建一个与fft_x1相同的tensor，用于存储融合后的频谱图
    fft_clone[:, :, :, :, 0] = torch.cos(pha_x1.clone()) * amp_x1_new.clone()
    fft_clone[:, :, :, :, 1] = torch.sin(pha_x1.clone()) * amp_x1_new.clone()

    # get the recomposed image: source content, target style  # 重构图像
    # _, _, imgH, imgW = x1.size()  # 获取图像的高和宽
    amp_pha_unwrap = torch.fft.ifft2(torch.complex(fft_clone[:, :, :, :, 0], fft_clone[:, :, :, :, 1]),
                                     dim=(-2, -1)).float()  # 对融合后的频谱图进行二维离散傅里叶逆变换
    return amp_pha_unwrap  # 返回重构图像


# 相位展开 F^(-1) [θe^(-jP(x) ) ]
def pha_unwrapping(x):  # x为频谱图
    fft_x = torch.fft.fft2(x.clone(), dim=(-2, -1))  # 对输入进行二维离散傅里叶变换
    fft_x = torch.stack((fft_x.real, fft_x.imag), dim=-1)  # 连接实部和虚部
    pha_x = torch.atan2(fft_x[:, :, :, :, 1], fft_x[:, :, :, :, 0])  # 反正切函数计算相位

    fft_clone = torch.zeros(fft_x.size(), dtype=torch.float).cuda()  # 创建一个与fft_x相同的张量
    fft_clone[:, :, :, :, 0] = torch.cos(pha_x.clone())
    fft_clone[:, :, :, :, 1] = torch.sin(pha_x.clone())

    # 获取重组后的图像:源内容，目标样式
    pha_unwrap = torch.fft.ifft2(torch.complex(fft_clone[:, :, :, :, 0], fft_clone[:, :, :, :, 1]),
                                 dim=(-2, -1)).float()  # 对输入进行二维离散傅里叶逆变换

    return pha_unwrap  # 返回相位展开后的图片
