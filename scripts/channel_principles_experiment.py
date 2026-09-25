"""A：用受控输入验证平滑抑噪和整体亮度平移的边界条件。"""
import argparse
import numpy as np
from src.channels11 import compute_11_channels
from src.data_io import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    rng=np.random.default_rng(42)
    # 常量真值128叠加零均值独立高斯噪声；它是机理实验，不是检测数据。
    noisy=np.clip(np.rint(128+rng.normal(0,20,(512,512))),0,255).astype(np.uint8)
    channels=compute_11_channels(noisy)
    variances=[float(channels[i][:512-m,:512-m].astype(float).var()) for i,m in enumerate((0,1,3))]
    base=rng.integers(20,180,(64,64),dtype=np.uint8)
    brighter=base+30  # 最高209，不发生uint8溢出或饱和。
    original=compute_11_channels(base);shifted=compute_11_channels(brighter)
    difference=[int(np.max(abs(a.astype(np.int16)-b.astype(np.int16)))) for a,b in zip(original[3:],shifted[3:])]
    saturated=compute_11_channels(np.clip(base.astype(np.int16)+150,0,255).astype(np.uint8))
    saturation_delta=max(int(np.max(abs(a.astype(np.int16)-b.astype(np.int16)))) for a,b in zip(original[3:],saturated[3:]))
    write_json(args.output,dict(seed=42,synthetic=True,experiment='controlled channel mechanism, not detection accuracy',
                              noise_variances_C0_C1_C2=variances,
                              variance_ratios=[value/variances[0] for value in variances],
                              global_brightness_plus30_gradient_max_difference=difference,
                              saturated_plus150_gradient_max_difference=saturation_delta,
                              caveat='Variance assumes independent noise on a flat field; real edges and correlated noise differ'))
    print('Noise variance ratios:',[round(v/variances[0],4) for v in variances],
          'brightness gradient error:',max(difference),'saturation error:',saturation_delta)


if __name__=='__main__': main()
