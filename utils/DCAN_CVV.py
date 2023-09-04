import torch
import torch.nn.functional as F


def ACM(cnn_out, img, mask, iter):
    epsilon = 1e-8
    batchsize = cnn_out.shape[0]
    height = cnn_out.shape[2]
    width = cnn_out.shape[3]

    CNN = torch.nn.Conv2d(3, 1, 3, 1, 1).cuda()
    img = CNN(img)

    two_tensor = torch.full(cnn_out.size(), 2.0).cuda()
    lambda0 = cnn_out*0.001

    Gx = torch.FloatTensor([-1, 1, 0]).cuda()
    filterx = torch.reshape(Gx, [1, 1, 3, 1])
    Gy = torch.FloatTensor([-1, 1, 0]).cuda()
    filtery = torch.reshape(Gy, [1, 1, 1, 3])
    laplace = torch.FloatTensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]]).cuda()
    filterlaplace = torch.reshape(laplace, [1, 1, 3, 3])

    dx = torch.zeros([batchsize, 1, height, width]).cuda()
    dy = torch.zeros([batchsize, 1, height, width]).cuda()
    bx = torch.zeros([batchsize, 1, height, width]).cuda()
    by = torch.zeros([batchsize, 1, height, width]).cuda()

    u = cnn_out
    c1 = 1
    c2 = 0
    for i in range(iter):
        # tf.reduce_sum()作用是按一定方式计算张量中元素之和
        lambda1 = torch.exp(torch.divide(torch.subtract(two_tensor, u), torch.add(1.0, u)))  # 内
        lambda2 = torch.exp(torch.divide(torch.add(1.0, u), torch.subtract(two_tensor, u)))  # 外
        # ----------------------update u------------------------#
        r = torch.multiply(lambda1, torch.square(u - c1)) - torch.multiply(lambda2, torch.square(mask - c2))
        bx_partial = F.conv2d(bx, filterx, stride=1, padding=[1, 0]).cuda()
        by_partial = F.conv2d(by, filtery, stride=1, padding=[0, 1]).cuda()
        dx_partial = F.conv2d(dx, filterx, stride=1, padding=[1, 0]).cuda()
        dy_partial = F.conv2d(dy, filtery, stride=1, padding=[0, 1]).cuda()
        alpha = bx_partial + by_partial - dx_partial - dy_partial
        temp = torch.multiply(torch.div(1, 0.5), r).cuda()
        beta = 0.25 * (F.conv2d(u, filterlaplace, stride=1, padding=1) + alpha - temp).cuda()
        u = beta

        # torch.greater 判断前者大于后者，满足Ture;否则False.
        high0 = torch.ones([batchsize, 1, height, width]).cuda()
        u = torch.where(torch.greater(u, high0), high0, u).cuda()
        low0 = torch.zeros([batchsize, 1, height, width]).cuda()
        u = torch.where(torch.greater(low0, u), low0, u).cuda()

        # ----------------------update d------------------------#
        Ix = F.conv2d(u, filterx, stride=1, padding=[1, 0]).cuda()
        Iy = F.conv2d(u, filtery, stride=1, padding=[0, 1]).cuda()
        # tf.abs()求绝对值
        tempx1 = torch.abs(Ix + bx) - torch.div(high0, lambda0 + epsilon)
        tempx1 = torch.where(torch.greater(low0, tempx1), low0, tempx1)
        tempx2 = torch.sign(Ix + bx)
        dx = torch.multiply(tempx1, tempx2)  # 相同位置的元素相乘

        tempy1 = torch.abs(Iy + by) - torch.div(high0, lambda0 + epsilon)
        tempy1 = torch.where(torch.greater(low0, tempy1), low0, tempy1)
        tempy2 = torch.sign(Iy + by)
        dy = torch.multiply(tempy1, tempy2)

        # ----------------------update b-------------------------#
        bx = bx + Ix - dx
        by = by + Iy - dy

        # ----------------------update C1,C2--------------------------#
        gamma0 = (torch.ones([batchsize, 1, height, width]) * 0.5).cuda()
        region_in = torch.where(torch.greater_equal(u, gamma0), high0, low0).cuda()
        region_out = torch.where(torch.less(u, gamma0), high0, low0)

        c1 = torch.sum(torch.multiply(region_in, img)) / (torch.sum(region_in) + epsilon)
        c2 = torch.sum(torch.multiply(region_out, img)) / (torch.sum(region_out) + epsilon)

    pred = u

    return pred
