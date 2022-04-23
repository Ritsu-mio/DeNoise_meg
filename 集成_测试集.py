import numpy as np
import sys

if __name__ == '__main__':
    pred_name = 'workspace/集成_1.bin'
    pred_name_filpx = 'workspace/集成_2.bin'
    pred_name_filpy = 'workspace/集成_3.bin'
    pred_name_filpx_filpy = 'workspace/集成_4.bin'

    content = open(pred_name, 'rb').read()
    samples_pred = np.float32(np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256)))

    content = open(pred_name_filpx, 'rb').read()
    samples_pred_filpx = np.float32(np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256)))

    content = open(pred_name_filpy, 'rb').read()
    samples_pred_filpy = np.float32(np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256)))

    content = open(pred_name_filpx_filpy, 'rb').read()
    samples_pred_filpx_filpy = np.float32(np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256)))


    samples_pred = samples_pred * 0.75
    samples_pred_filpx = samples_pred_filpx * 0.15
    samples_pred_filpy = samples_pred_filpy * 0.1
    samples_pred_filpx_filpy = samples_pred_filpx_filpy * 0.00
    samples_asem = samples_pred + samples_pred_filpx + samples_pred_filpy + samples_pred_filpx_filpy

    fout = open('workspace/test_集成.bin', 'wb')
    samples_asem = samples_asem.astype('uint16')
    fout.write(samples_asem.tobytes())
    fout.close()
