"""A：批量生成通道图和整数/浮点对照；输入清单明确记录图片类别与来源。"""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from src.channels11 import compute_11_channels,compute_11_channels_float
from src.data_io import read_image,write_image,write_json
from .visualize_channels import channel_grid

# 通道右侧/下侧的无效宽度，计算差异时不让补零边框影响结果。
MARGINS = [0,1,3,3,3,3,3,7,7,7,7]


def run(manifest,output):
    """读取观察样本；保存单通道原尺寸图、拼图、差异和非基准级耗时。"""
    manifest,output = Path(manifest),Path(output)
    samples = json.loads(manifest.read_text(encoding='utf-8'))
    identifiers = [item['id'] for item in samples]
    if len(identifiers)!=len(set(identifiers)) or any(Path(s).name!=s or not s for s in identifiers):
        raise ValueError('Sample IDs must be unique simple filenames')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new experiment output folder')
    records = []
    for item in samples:
        gray = cv2.cvtColor(read_image(manifest.parent/item['image']),cv2.COLOR_BGR2GRAY)
        integer = compute_11_channels(gray)
        quantized = compute_11_channels_float(gray,True)
        continuous = compute_11_channels_float(gray,False)
        folder=output/item['id']
        write_image(folder/'grid.png',channel_grid(gray))
        write_image(folder/'gray.png',gray)
        details=[]
        for index,(plane,q,f,margin) in enumerate(zip(integer,quantized,continuous,MARGINS)):
            # 固定命名，便于把结果逐张对照到 C0～C10 的公式。
            write_image(folder/f'C{index:02d}.png',plane)
            h,w=gray.shape[0]-margin,gray.shape[1]-margin
            if min(h,w)<=0:
                raise ValueError('Experiment images must be at least 8x8')
            delta=abs(plane[:h,:w].astype(float)-f[:h,:w])
            exact=float(np.max(abs(plane.astype(float)-q)))
            if exact != 0:
                raise AssertionError('Same-rounding float and integer implementations differ')
            details.append(dict(channel=index,valid_pixels=h*w,integer_float_quantized_max_difference=exact,
                                unrounded_max_difference=float(delta.max()),unrounded_mean_difference=float(delta.mean())))
        timing={}
        for name,func in [('integer',compute_11_channels),('float_same_rounding',compute_11_channels_float)]:
            func(gray)  # 预热一次，再取七次中位数，减少首次调用的影响。
            elapsed=[]
            for _ in range(7):
                start=time.perf_counter(); func(gray); elapsed.append((time.perf_counter()-start)*1000)
            timing[name]=float(np.median(elapsed))
        records.append(dict(**item,width=gray.shape[1],height=gray.shape[0],channels=details,
                            median_ms=timing,timing_note='local illustrative measurements; not a benchmark'))
    write_json(output/'measurements.json',records)
    lines=['# 样本的 11 通道实验','',
           '所有图片按 0～255 原值显示；右侧和下侧黑边是无效边界补零，零梯度约为中灰。',
           '浮点同取整用于验证实现；不取整浮点用于观察量化差异，不能直接替换已训练模型的通道。','']
    for item in records:
        lines += [f'## {item["id"]}：{item["category"]}','',f'来源：{item["source"]}',
                  '',f'处理说明：{item.get("preparation","原尺寸灰度化")}',
                  '',f'![原图与11通道]({item["id"]}/grid.png)','']
    (output/'index.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Saved {len(records)} channel experiments to {output}')


def main():
    """命令行传入观察清单和新的输出目录。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run(args.manifest,args.output)


if __name__=='__main__':
    main()
