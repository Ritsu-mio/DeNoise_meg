import megengine as mge
import megengine.module as M
import megengine.functional as F
import random
import numpy as np
from megengine.utils.module_stats import module_stats
from swinir_meg_2_Dense import Predictor
#from torch.utils.tensorboard import SummaryWriter
from dataset import build_dataset_all
import tqdm
import time
import math
patchsz = 256
batchsz = 1
#tb_logger = SummaryWriter("runs/swinir_meg_2_Dense_bs1_ws16_L1")

# class Predictor(M.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = M.Sequential(
#             M.Conv2d(4, 50, 3, padding = 1, bias = True),
#             M.LeakyReLU(negative_slope = 0.125),
#             M.Conv2d(50, 50, 3, padding = 1, bias = True),
#             M.LeakyReLU(negative_slope = 0.125),
#         )
#         self.conv2 = M.Sequential(
#             M.Conv2d(50, 50, 3, padding = 1, bias = True),
#             M.LeakyReLU(negative_slope = 0.125),
#             M.Conv2d(50, 50, 3, padding = 1, bias = True),
#             M.LeakyReLU(negative_slope = 0.125),
#         )
#         self.conv3 = M.Sequential(
#             M.Conv2d(50, 50, 3, padding = 1, bias = True),
#             M.LeakyReLU(negative_slope = 0.125),
#             M.Conv2d(50, 4, 3, padding = 1, bias = True),
#             M.LeakyReLU(negative_slope = 0.125),
#         )
#     def forward(self, x):
#         n, c, h, w = x.shape
#         x = x.reshape((n, c, h // 2, 2, w // 2, 2)).transpose((0, 1, 3, 5, 2, 4)).reshape((n, c * 4, h // 2, w // 2))
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = x.reshape((n, c, 2, 2, h // 2, w // 2)).transpose((0, 1, 4, 2, 5, 3)).reshape((n, c, h, w))
#         return x


class CharbonnierLoss(M.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = F.mean(F.sqrt(diff * diff + self.eps))
        return loss

def cosine(lr_min, lr_max, per_epochs,epoch):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(epoch / per_epochs * np.pi))

def train():


    #net = Predictor()
    #net.load_state_dict(mge.load("./model/twin_net_IMDB_2_4channel/1399999_model"))
    
    height = 64
    width = 64
    window_size = 16
    in_chans = 1
    net = Predictor(
        upscale=1,
        img_size=(height, width),
        window_size=window_size,
        img_range=1.,
        depths=[6, 4, 4],
        embed_dim=24,
        in_chans=in_chans,
        num_heads=[4, 4, 4],
        mlp_ratio=2,
        upsampler=None)
    
#     for m in net.modules():
#         if isinstance(m, M.Conv2d):
#             M.init.msra_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#             if m.bias is not None:
#                 fan_in, _ = M.init.calculate_fan_in_and_fan_out(m.weight)
#                 bound = 1 / math.sqrt(fan_in)
#                 M.init.uniform_(m.bias, -bound, bound)
#         elif isinstance(m, M.BatchNorm2d):
#             M.init.ones_(m.weight)
#             M.init.zeros_(m.bias)
#         elif isinstance(m, M.Linear):
#             M.init.msra_uniform_(m.weight, a=math.sqrt(5))
#             if m.bias is not None:
#                 fan_in, _ = M.init.calculate_fan_in_and_fan_out(m.weight)
#                 bound = 1 / math.sqrt(fan_in)
#                 M.init.uniform_(m.bias, -bound, bound)
    
    input_data = np.random.rand(1, 1, 256, 256).astype("float32")
    total_stats, stats_details = module_stats(
        net,
        inputs=(input_data,),
        cal_params=True,
        cal_flops=True,
        logging_to_stdout=True,
    )

    print("params %.3fK MAC/pixel %.0f" % (
        total_stats.param_dims / 1e3, total_stats.flops / input_data.shape[2] / input_data.shape[3]))



    train_steps = 4600002
    learning_rate = 1e-3
    learning_rate_lest = 1e-7
    opt = mge.optimizer.Adam(net.parameters(), lr=learning_rate)
    #opt.load_state_dict(mge.load("./model/twin_net_IMDB_2_4channel/1399999_opt"))
    gm = mge.autodiff.GradManager().attach(net.parameters())
    cb_loss = CharbonnierLoss()
    losses = []

    rnd = random.Random(100)

    print('training')

    train_dataloader,val_dataloader= build_dataset_all(batch_size=batchsz,workers=0,lr_path='dataset/competition_train_input.0.2.bin',gt_path='dataset/competition_train_gt.0.2.bin',crop_size=64)
    train_queue = iter(train_dataloader)
    #print("train_queue=",train_queue)
    
    step_k = 0
    Step_epoch = [40000,80000,160000,320000,640000]
    Step_sum = [0,40000, 120000, 280000, 600000, 1240000]
    for it in range(0, train_steps):
#         current_lr = learning_rate * (Step_sum[step_k] - it) / Step_epoch[step_k]
#         if it==1400000:
#             learning_rate = 1e-4
#             learning_rate_lest = 1e-8
        if (Step_sum[step_k+1] == it):
            step_k = step_k + 1
        #current_lr = learning_rate * (Step_sum[step_k] - it) / Step_epoch[step_k]
        current_lr = cosine(learning_rate_lest, learning_rate, Step_epoch[step_k], it-Step_sum[step_k])
        for g in opt.param_groups:
            g['lr'] = current_lr

        opt.clear_grad()
        #opt.clear_grad()
        #batch_inp = next(train_queue)
        batch_inp, batch_out = next(train_queue)
        batch_inp = mge.tensor(batch_inp)
        batch_out = mge.tensor(batch_out)

        #print("batch_inp=",batch_inp.shape)

        with gm:
            pred = net(batch_inp)
            loss = F.abs(pred - batch_out).mean()
            #loss = cb_loss(pred,batch_out)
            gm.backward(loss)
            opt.step().clear_grad()

        loss = float(loss.numpy())
        losses.append(loss)
        if it % 1000 == 0:
            print('it', it, 'loss', loss, 'mean', np.mean(losses[-100:]), "time",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            #tb_logger.add_scalar("mean_loss", np.mean(losses[-100:]), it)
            #tb_logger.add_scalar("loss", loss, it)
            file = open('./model/swinir_meg_2_Dense_bs1_ws16_L1_filp/log.txt','a')
            file.write('it '+ str(it)+ ',loss '+ str(loss)+ ',mean '+ str(np.mean(losses[-100:]))+',time'+str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))+ '\n')
            file.close()
        if (it+1) % 5000 == 0:
            fout = open("./model/swinir_meg_2_Dense_bs1_ws16_L1_filp/" + str(it) + "_model", 'wb')
            mge.save(net.state_dict(), fout)
            fout.close()
            fout1 = open("./model/swinir_meg_2_Dense_bs1_ws16_L1_filp/" + str(it) + "_opt", 'wb')
            mge.save(opt.state_dict(), fout1)
            fout1.close()
        if (it+1) % 10000 ==0:
            net.eval()
            pred_s = []
            gt_s = []
            for i, (lr_img, gt_img) in enumerate(val_dataloader):
                lr_img = mge.tensor(lr_img)
                gt_img = mge.tensor(gt_img)
                pred = net(lr_img)
                pred = (pred.numpy()[:, 0, :, :] * 65536).clip(0, 65535).astype('uint16')
                gt_img = (gt_img.numpy()[:, 0, :, :] * 65536).clip(0, 65535).astype('uint16')
                pred_s.append(pred)
                gt_s.append(gt_img)
            pred_s = np.float32(np.array(pred_s)[:,0])
            gt_s = np.float32(np.array(gt_s)[:,0])
            means = gt_s.mean(axis=(1, 2))
            weight = (1 / means) ** 0.5
            diff = np.abs(pred_s - gt_s).mean(axis=(1, 2))
            diff = diff * weight
            score = diff.mean()
            # print("sss=",score)
            score = np.log10(100 / score) * 5
            #tb_logger.add_scalar("val_score", score, it)
            print('score', score)
            file = open('./model/swinir_meg_2_Dense_bs1_ws16_L1_filp/log.txt','a')
            file.write('score '+ str(score)+ '\n')
            file.close()
            net.train()

def test():
    #content = open('dataset/competition_train_input.0.2.bin', 'rb').read()
    content = open('dataset/competition_test_input.0.2.bin', 'rb').read()
    samples_ref = np.frombuffer(content, dtype = 'uint16').reshape((-1,256,256))
    fout = open('workspace/office_test.bin', 'wb')
    net = Predictor()
    net.load_state_dict(mge.load("./model/office_区分验证集/19000_model"))
    net.eval()
    import tqdm
    print("len(samples_ref)=",len(samples_ref))
    for i in tqdm.tqdm(range(0, len(samples_ref), batchsz)):
    #for i in tqdm.tqdm(range(0, 1000, batchsz)):
        i_end = min(i + batchsz, len(samples_ref))
        batch_inp = mge.tensor(np.float32(samples_ref[i:i_end, None, :, :]) * np.float32(1 / 65536))
        pred = net(batch_inp)
        pred = (pred.numpy()[:, 0, :, :] * 65536).clip(0, 65535).astype('uint16')
        fout.write(pred.tobytes())

    fout.close()

def test_val():
    content = open('dataset/competition_train_input.0.2.bin', 'rb').read()
    #content = open('dataset/competition_test_input.0.2.bin', 'rb').read()
    samples_ref = np.frombuffer(content, dtype = 'uint16').reshape((-1,256,256))
    fout = open('workspace/DR_net2_val.bin', 'wb')
    net = Predictor()
    net.load_state_dict(mge.load("./model/DR_net2/19000_model"))
    net.eval()
    import tqdm
    print("len(samples_ref)=",len(samples_ref))
    #for i in tqdm.tqdm(range(0, len(samples_ref), batchsz)):
    for i in tqdm.tqdm(range(0, 1000, batchsz)):
        i_end = min(i + batchsz, len(samples_ref))
        batch_inp = mge.tensor(np.float32(samples_ref[i:i_end, None, :, :]) * np.float32(1 / 65536))
        pred = net(batch_inp)
        pred = (pred.numpy()[:, 0, :, :] * 65536).clip(0, 65535).astype('uint16')
        fout.write(pred.tobytes())

    fout.close()

if __name__ == '__main__':
    #test_val()
    train()
