import megengine as mge
import megengine.module as M
import megengine.functional as F
import random
import numpy as np
from megengine.utils.module_stats import module_stats
from model_meg import Predictor


import time
import math
patchsz = 256
batchsz = 16



def test():
    #content = open('dataset/competition_train_input.0.2.bin', 'rb').read()
    content = open('dataset/competition_test_input.0.2.bin', 'rb').read()
    samples_ref = np.frombuffer(content, dtype = 'uint16').reshape((-1,256,256))

    fout = open('workspace/test.bin', 'wb')

    height = 256
    width = 256
    window_size = 16
    net = Predictor(
        upscale=1,
        img_size=(height, width),
        window_size=window_size,
        img_range=1.,
        depths=[4, 4],
        embed_dim=32,
        in_chans=1,
        num_heads=[4, 4],
        mlp_ratio=2,
        upsampler=None)

    net.load_state_dict(mge.load("./model/test_model"))
    net.eval()

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

    import tqdm
    print("len(samples_ref)=",len(samples_ref))
    batchsz = 8
    for i in tqdm.tqdm(range(0, len(samples_ref), batchsz)):
    #for i in tqdm.tqdm(range(0, 1000, batchsz)):
        i_end = min(i + batchsz, len(samples_ref))
        batch_inp = mge.tensor(np.float32(samples_ref[i:i_end, None, :, :]) * np.float32(1 / 65536))
        pred = net(batch_inp)
        pred = (pred.numpy()[:, 0, :, :] * 65536).clip(0, 65535).astype('uint16')

        fout.write(pred.tobytes())

    fout.close()



if __name__ == '__main__':

    test()