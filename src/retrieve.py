# Copyright 2022 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import argparse
import os
import tqdm
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
from clip_retrieval.clip_client import ClipClient


def retrieve(target_name, outpath, num_class_images):
    '''
    1: IDK why there is 2* ...

    '''
    num_images = 2 * num_class_images
    client = ClipClient(url="https://knn.laion.ai/knn-service", indice_name="laion_400m", num_images=num_images,
                        aesthetic_weight=0.1)  # TODO COMMENT: this clipclient fetches images based on datasets from clip when you query it with some text prompt

    if len(target_name.split()):
        target = '_'.join(target_name.split())
    else:
        target = target_name
    os.makedirs(f'{outpath}/{target}', exist_ok=True)

    if len(list(Path(f'{outpath}/{target}').iterdir())) >= num_class_images:
        return

    while True:
        results = client.query(
            text=target_name)  # TODO COMMENT: this is the part of querying the clip client which given text or image/s, search for other captions/images that are semantically similar.
        if len(results) >= num_class_images or num_images > 1e4:
            break
        else:
            num_images = int(1.5 * num_images)
            client = ClipClient(url="https://knn.laion.ai/knn-service", indice_name="laion_400m", num_images=num_images,
                                aesthetic_weight=0.1)

    count = 0
    urls = []
    captions = []

    pbar = tqdm.tqdm(desc='downloading real regularization images', total=num_class_images)

    count = gather_images_urls_and_captions_in_lists_and_stop_when_getting_to_the_exact_number(captions, count,
                                                                                               num_class_images,
                                                                                               outpath, pbar, results,
                                                                                               target, urls)

    write_all_captions_urls_and_imageNumbers_in_txt_files(captions, count, outpath, target, urls)


def write_all_captions_urls_and_imageNumbers_in_txt_files(captions, count, outpath, target, urls):
    with open(f'{outpath}/caption.txt', 'w') as f:
        for each in captions:
            f.write(each.strip() + '\n')  # TODO COMMENT: strip() removes spaces from a string.
    with open(f'{outpath}/urls.txt', 'w') as f:
        for each in urls:
            f.write(each.strip() + '\n')  # TODO COMMENT: strip() removes spaces from a string.
    with open(f'{outpath}/images.txt', 'w') as f:
        for p in range(count):
            f.write(f'{outpath}/{target}/{p}.jpg' + '\n')  # TODO COMMENT: strip() removes spaces from a string.


def gather_images_urls_and_captions_in_lists_and_stop_when_getting_to_the_exact_number(captions, count,
                                                                                       num_class_images, outpath, pbar,
                                                                                       results, target, urls):
    for each in results:
        name = f'{outpath}/{target}/{count}.jpg'
        success = True
        while True:
            try:
                img = requests.get(each['url'])
                success = True
                break
            except:
                success = False
                break
        if success and img.status_code == 200:
            try:
                _ = Image.open(BytesIO(img.content))
                with open(name, 'wb') as f:
                    f.write(img.content)
                urls.append(each['url'])
                captions.append(each['caption'])
                count += 1
                pbar.update(1)
            except:
                pass
        if count > num_class_images:
            break
    return count


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--target_name', help='target string for query',
                        type=str)
    parser.add_argument('--outpath', help='path to save retrieved images', default='./',
                        type=str)
    parser.add_argument('--num_class_images', help='number of retrieved images', default=200,
                        type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    retrieve(args.target_name, args.outpath,
             args.num_class_images)  # TODO COMMENT: I think the num_class_images is for the regularization.
