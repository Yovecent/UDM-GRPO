# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------

import json
import os
import subprocess

import codewithgpu
import tqdm


def get_jsonl_iter(path, use_tqdm=True):
    total = int(subprocess.check_output(f"wc -l {path}", shell=True).strip().split()[0])
    return tqdm.tqdm(open(path), total=total, disable=not use_tqdm)


if __name__ == "__main__":

    jsonl_path = os.path.join(os.path.dirname(__file__), "train_metadata.jsonl")
    record_path = "./geneval_50k"

    # jsonl_path = os.path.join(os.path.dirname(__file__), "test_metadata.jsonl")
    # record_path = "./geneval_2k"

    os.makedirs(record_path)
    features = {"id": "string", "caption": "string", "text": "string", "metadata": "string"}
    writer = codewithgpu.RecordWriter(record_path, features, zfill_width=6)
    for i, data in enumerate(get_jsonl_iter(jsonl_path)):
        caption = json.loads(data)["prompt"]
        example = {"id": str(i).zfill(8), "text": caption, "caption": caption}
        writer.write({**{"metadata": data.strip()}, **example})
    writer.close()
