"""A：准备六张通道观察图；人像路径显式传入，不依赖某台电脑的安装位置。"""
import argparse
from pathlib import Path
import cv2
from src.data_io import read_image, write_image, write_json


def main():
    """使用指定的 Hopper 和 astronaut 原图，以及本地课程数据集。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hopper', required=True, help='Matplotlib 样例 grace_hopper.jpg')
    parser.add_argument('--astronaut', required=True, help='scikit-image 样例 astronaut.png')
    parser.add_argument('--anime-root', default='data/animeface/images')
    parser.add_argument('--manga-root', default='data/Manga109_released_2026_05_21')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new observation output folder')
    # 两张真实照片使用固定观察区域；这些裁剪框不进入训练真值。
    specs = [
        ('real_hopper', '真实近正面人脸', Path(args.hopper), [135,105,395,375],
         'Matplotlib grace_hopper.jpg; https://matplotlib.org/stable/gallery/images_contours_and_fields/image_demo.html'),
        ('real_astronaut', '真实近正面人脸', Path(args.astronaut), [145,25,300,195],
         'scikit-image NASA astronaut; https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut'),
        ('anime_frontal_1', '近正面动漫人脸', Path(args.anime_root)/'16572_2006.jpg', None, 'AnimeFace: 16572_2006.jpg'),
        ('anime_frontal_2', '近正面动漫人脸', Path(args.anime_root)/'24459_2009.jpg', None, 'AnimeFace: 24459_2009.jpg'),
        ('anime_hair_occlusion', '头发遮挡动漫人脸', Path(args.anime_root)/'0_2000.jpg', None, 'AnimeFace: 0_2000.jpg'),
        ('manga_complex', '复杂漫画页面', Path(args.manga_root)/'images/AisazuNihaIrarenai/040.jpg', None,
         'Manga109-v2026: AisazuNihaIrarenai/040.jpg')]
    samples = []
    for name, category, path, box, source in specs:
        image = read_image(path)
        preparation = '使用原图'
        if box:
            x1,y1,x2,y2 = box
            if x2 > image.shape[1] or y2 > image.shape[0]:
                raise ValueError(f'Portrait size does not match expected source: {path}')
            image = image[y1:y2,x1:x2]
            preparation = f'固定观察区 xyxy={box}，不是真值标注'
        if name == 'manga_complex':
            scale = 512/max(image.shape[:2])
            image = cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)))
            preparation = '完整双页按比例缩到最长边512'
        write_image(output/f'{name}.png', image)
        samples.append(dict(id=name, category=category, image=f'{name}.png', source=source, preparation=preparation))
    write_json(output/'samples.json', samples)
    print(f'Prepared {len(samples)} observation images in {output}')


if __name__ == '__main__':
    main()
