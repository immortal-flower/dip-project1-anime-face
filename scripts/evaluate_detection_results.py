"""A：评价 B 已导出的整页检测结果，负责统计，不负责训练或生成预测。"""
import argparse
import json
from pathlib import Path
from src.data_io import write_json
from src.detection_metrics import evaluate_dataset


def main():
    """预测 JSON 格式为 {page_id: [{bbox:[x1,y1,x2,y2],score:数值}, ...]}。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pages',required=True)
    parser.add_argument('--predictions',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--split',choices=['train','val','test'],default='test')
    parser.add_argument('--iou',type=float,default=.5)
    args=parser.parse_args()
    pages=json.loads(Path(args.pages).read_text(encoding='utf-8'))
    selected=[page for page in pages if page['split']==args.split]
    # 一个预测文件只评价指定集合；不静默丢弃来自其他图片的错误编号。
    predictions=json.loads(Path(args.predictions).read_text(encoding='utf-8'))
    if not isinstance(predictions,dict):
        raise ValueError('Predictions must be a JSON object keyed by page_id')
    report=evaluate_dataset(selected,predictions,args.iou)
    report['split']=args.split
    write_json(args.output,report)
    print({key:report[key] for key in ('page_count','tp','fp','fn','precision','recall','f1')})


if __name__=='__main__':
    main()
