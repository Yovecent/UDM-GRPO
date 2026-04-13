# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""PickScore evalution."""

import argparse
import json
import os

import torch
import tqdm
import PIL.Image

from diffnext.rewards.reward_image import PickScoreReward


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="pickscore evaluation")
    parser.add_argument("--image_root", type=str, default="", help="image root")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--out_file", type=str, default="", help="result file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.out_file = args.out_file if args.out_file else args.image_root + ".json"

    device = torch.device("cuda", 0)
    data_paths = [os.path.join(args.image_root, _) for _ in os.listdir(args.image_root)]
    data_paths.sort()

    all_images, all_prompts, all_rewards = [], [], []
    for data_path in data_paths:
        images = [os.path.join(data_path, "samples", _) for _ in os.listdir(data_path + "/samples")]
        prompts = [json.load(open(data_path + "/metadata.jsonl"))["prompt"]] * len(images)
        all_images.extend(images), all_prompts.extend(prompts)

    model = PickScoreReward(batch_size=args.batch_size).to(device)
    data_iter = list(map(model.batch_iter, (all_images, all_prompts)))
    for images, prompts in tqdm.tqdm(zip(*data_iter), total=len(data_iter[0])):
        images, prompts = [PIL.Image.open(_) for _ in images], list(prompts)
        all_rewards += model.compute_pickscore(images, prompts)["rewards"]

    mean_score = sum(all_rewards) / len(all_rewards)
    print("PickScore =", mean_score)
    result_dict = {"pickscore": [mean_score, []]}
    for x, y, z in zip(all_images, all_prompts, all_rewards):
        result_dict["pickscore"][-1].append({"image_path": x, "prompt": y, "score": z})
    json.dump(result_dict, open(args.out_file, "w"))
