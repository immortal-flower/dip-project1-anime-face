"""A：导出本地阶段交接包并逐文件生成校验清单；不包含完整原始漫画库。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from src.data_io import write_json


def export(repo,dataset,channels,output):
    repo,dataset,channels,output=map(lambda p:Path(p).resolve(),(repo,dataset,channels,output))
    if output.exists(): raise FileExistsError('Use a new delivery directory')
    # 保持data/processed/版本名层次，使样本清单里的../../原始数据路径仍可解释。
    relative=dataset.relative_to(repo)
    if relative.parts[:2] != ('data','processed'):
        raise ValueError('Dataset must be inside repository data/processed')
    output.mkdir(parents=True)
    for folder in ('src','scripts','tests','docs','configs'):
        for path in (repo/folder).rglob('*'):
            if path.is_file() and path.suffix in ('.py','.md','.json','.pdf') and '__pycache__' not in path.parts:
                target=output/path.relative_to(repo);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    for name in ('README.md','requirements.txt','requirements-lock.txt','demo.py','AGENTS.md'):
        if (repo/name).exists(): shutil.copy2(repo/name,output/name)
    shutil.copytree(dataset,output/relative)
    shutil.copytree(channels,output/'results'/channels.name)
    if channels.name!='channel_experiment_v1' and (repo/'results/channel_experiment_v1').exists():
        shutil.copytree(repo/'results/channel_experiment_v1',output/'results/channel_experiment_v1')
    for folder in ('quality_review_v2','near_duplicates_manga_v2','near_duplicates_anime_v1','environment_a'):
        if (repo/'results'/folder).exists(): shutil.copytree(repo/'results'/folder,output/'results'/folder)
    for name in ('a_data_audit_v2.json','crop_version_validation.json','channel_principles.json'):
        if (repo/'results'/name).exists(): shutil.copy2(repo/'results'/name,output/'results'/name)
    (output/'START_HERE.md').write_text(
        '# A 阶段交接包（候选版）\n\n'
        '先读 docs/A_WORK_LOG.md 和 docs/A_REPORT.md。数据尚未全量人工验收，不是最终课程提交包。\n\n'
        '包含代码、配置、24×24样本、固定划分、原真值引用、通道图、质量与重复候选记录、环境证据。\n'
        '未包含完整 Manga109 页面、全量 AnimeFace、C标注和真实模型。复核原图及整页评价需按来源获取原数据，'
        '放到 data/Manga109_released_2026_05_21 和 data/animeface。\n\n'
        '原图复核前可读取 data/processed/'+dataset.name+'/manifest_candidate.json 进行接口实验；'
        '该清单仍为 needs_review，不能把数据筛选结果冒充最终测试准确率。\n\n'
        'SHA256SUMS.json 记录包内内容，可逐文件核验。本包只生成在本地，公开Git仓库不含图像数据。\n',encoding='utf-8')
    records=[]
    for path in sorted(output.rglob('*')):
        if path.is_file(): records.append(dict(path=path.relative_to(output).as_posix(),bytes=path.stat().st_size,
                                               sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write_json(output/'SHA256SUMS.json',records)
    # 写后再读一次，防止复制或磁盘写入过程中缺文件。
    for row in records:
        assert hashlib.sha256((output/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    print(f'Exported and verified {len(records)} files; {sum(r["bytes"] for r in records)} bytes')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True);parser.add_argument('--channels',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();export(Path(__file__).resolve().parents[1],args.dataset,args.channels,args.output)


if __name__=='__main__': main()
