"""A：保存真实环境、测试和合成流程输出，生成可查看的环境核验页面。"""
import argparse
import html
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from src.data_io import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uv',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists(): raise FileExistsError('Use a new environment record folder')
    output.mkdir(parents=True)
    commands=[['uv_version',[str(Path(args.uv).resolve()),'--version']],
              ['tests',[sys.executable,'-X','utf8','-m','unittest','discover','-s','tests','-v']],
              ['synthetic_smoke',[sys.executable,'-X','utf8','-m','scripts.smoke_test','--output-dir',str(output/'smoke')]]]
    records=[]
    environment=dict(os.environ,PYTHONUTF8='1')
    for name,command in commands:
        result=subprocess.run(command,capture_output=True,env=environment)
        text=(result.stdout+result.stderr).decode('utf-8',errors='replace')
        (output/f'{name}.log').write_text(text,encoding='utf-8')
        records.append(dict(name=name,command=command,returncode=result.returncode,output=text))
    report=dict(python=sys.version,executable=sys.executable,numpy=version('numpy'),
                opencv=version('opencv-python'),checks=records,ok=all(r['returncode']==0 for r in records),
                note='Real command output; synthetic smoke is not a real-data accuracy result')
    write_json(output/'environment.json',report)
    blocks=''.join(f'<h2>{html.escape(r["name"])}</h2><pre>{html.escape(r["output"])}</pre>' for r in records)
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>A 环境核验记录</title><style>body{font:16px system-ui;max-width:1000px;margin:32px auto;color:#173042;background:#f4f7fa}h1{font-size:28px}h2{font-size:20px}pre{background:white;border:1px solid #cbd5df;border-radius:8px;padding:16px;white-space:pre-wrap;font:13px Consolas,monospace}p{line-height:1.6}</style>'
    page+=f'<h1>A 部分：独立环境核验</h1><p>实际执行结果：{"全部通过" if report["ok"] else "有失败，请查看日志"}。记录日期：2026-09-26。</p><p>Python {sys.version.split()[0]} · NumPy {report["numpy"]} · OpenCV {report["opencv"]}<br>uv 在项目工具目录；Python 与依赖安装于独立环境。下面保留真实输出，合成流程不代表真实检测准确率。</p>'+blocks
    (output/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('python','numpy','opencv','ok')},ensure_ascii=False))
    if not report['ok']: raise SystemExit(1)


if __name__=='__main__': main()
