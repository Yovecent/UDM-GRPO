<div align="center">

<h1>UDM-GRPO: Stable and Efficient Group Relative Policy Optimization for Uniform Discrete Diffusion Models</h1>

<p align="center">
<a href="https://arxiv.org/abs/2604.18518"><img src="https://img.shields.io/badge/ArXiv-2604.18518-%23840707.svg" alt="ArXiv"></a>
<a href="https://huggingface.co/collections/Yovecents/ursa-17b-ibq512-udm-grpo"><img src="https://img.shields.io/badge/🤗 Weights-UDMGRPO-rgb(166,109,59).svg" alt=""></a>
<a href="https://yovecent.github.io/UDM-GRPO.github.io/"><img src="https://img.shields.io/badge/Project-UDMGRPO-%237CB4F7.svg" alt="Project"></a>
</p>

[Jiaqi Wang](https://scholar.google.com/citations?user=EFOtaJsAAAAJ&hl=zh-CN)<sup>1,2*</sup>, [Haoge Deng](https://scholar.google.com/citations?user=S2sbvjgAAAAJ&hl=en)<sup>2*</sup>, [Ting Pan](https://scholar.google.com/citations?&user=qQv6YbsAAAAJ)<sup>2*</sup>,  [Yang Liu](https://scholar.google.com/citations?user=9JcQ2hwAAAAJ&hl)<sup>2</sup>, [Chengyuan Wang](https://scholar.google.co.uk/citations?user=LnMpl_wAAAAJ&hl=no)<sup>2</sup>, [Fan Zhang](https://scholar.google.com/citations?user=VsJ39HMAAAAJ)<sup>2</sup>, [Yonggang Qi](https://scholar.google.com.tw/citations?user=pQNpf7cAAAAJ&hl=zh-CN&oi=ao)<sup>1†</sup>, [Xinlong Wang](https://scholar.google.com/citations?user=DPz0DjYAAAAJ&hl)<sup>2†</sup><br>
<!-- [Chunhua Shen](https://scholar.google.com/citations?user=Ljk2BvIAAAAJ&hl)<sup>3</sup>, [Shiguang Shan](https://scholar.google.com/citations?user=Vkzd7MIAAAAJ&hl)<sup>2</sup>, [Zhaoxiang Zhang](https://scholar.google.com/citations?user=qxWfV6cAAAAJ&hl)<sup>1†</sup>, [Xinlong Wang](https://scholar.google.com/citations?user=DPz0DjYAAAAJ&hl)<sup>4†</sup><br> -->

[BUPT](https://www.bupt.edu.cn/)<sup>1</sup>, [BAAI](https://www.baai.ac.cn/en)<sup>2</sup><br>
<sup>*</sup> Equal Contribution, <sup>†</sup> Corresponding Author
<br><br><image src="assets/UDM-GRPO_pipeline.png"/>
<br><br><image src="assets/GenEval_result.png"/>
</div>

We propose **UDM-GRPO**, the first framework to integrate UDM with RL. Our method is guided by two key insights: (i) treating the final clean sample as the action provides more accurate and stable optimization signals; and (ii) reconstructing trajectories via the diffusion forward process better aligns probability paths with the pretraining distribution. Additionally, we introduce two strategies, Reduced-Step and CFG-Free, to further improve training efficiency. **UDM-GRPO** significantly improves base model, [URSA](https://github.com/baaivision/URSA?tab=readme-ov-file), performance across multiple T2I tasks. 

## 🚀 News

- ```[April 2026]``` Released [Paper](https://arxiv.org/abs/2604.18518) & [Project Page](https://yovecent.github.io/UDM-GRPO.github.io/) & [Model Weights](https://huggingface.co/collections/Yovecents/ursa-17b-ibq512-udm-grpo).

## ✨Hightlights

- 🥇 **Novel Approach**: Correcting the action and trajectory to achieve the first method to integrate UDM with GRPO.
- 🥈 **SOTA Performance**: State-of-the-art performance across multiple T2I benchmarks.
- 🥉 **High efficiency**: Reduced-Step and CFG-Free training strategy.

## 🤗 Model
| Task    | Model |
| -------- | -------- |
| GenEval     | [🤗GenEval](https://huggingface.co/Yovecents/URSA-1.7B-IBQ512-UDMGRPO-GenEval) |
| PickScore    | [🤗PickScore](https://huggingface.co/Yovecents/URSA-1.7B-IBQ512-UDMGRPO-PickScore) |

## 📖 Table of Contents
- [🔧 Installation](#installation)
- [🥫 Data Preparation](#datapreparation)
- [🤖 Training](#training)
- [🖋️ Evaluation](#evaluation)


## 🔧 Installation
<a id="installation"></a>

### 1. Environment Set Up
Clone this repository to local disk and install:
```bash
git clone https://github.com/Yovecent/UDM-GRPO.git

cd UDM-GRPO

conda create -n UDMGRPO python=3.10

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

pip install -e .

pip install torch==2.5.1 xformers --index-url https://download.pytorch.org/whl/cu124

pip install psutil==7.0.0, flash-attn==2.7.4.post1 --no-build-isolation
```


### 2. Model Download

| Model | Resolution| Download |
|:-----:|:----------:|:------:|
| URSA-1.7B-IBQ512 | 512x512 | [🤗 Hugging Face](https://huggingface.co/BAAI/URSA-1.7B-IBQ512)|



### 3. Reward Preparation



#### 1.PickScore
You can run the training code to download the [PickScore Model](https://huggingface.co/yuvalkirstain/PickScore_v1) or Pre-download.

#### 2. GenEval

##### Pip and download the mask2former
```bash
# First
pip install openmim==0.3.9 open-clip-torch==2.31.0 numpy==1.26.0 opencv-python==4.11.0.86 clip-benchmark==1.6.1


# Then
mim install mmengine mmcv-full==1.7.2 --no-build-isolation
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
pip install setuptools==78.1.1
pip install -e . --no-build-isolation


# Then
mv ../raw_rl_data/object_names.txt .

wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth \
-O ./mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth

```
#####  download the [timm/vit_large_patch14_clip_224.openai🤗](https://huggingface.co/timm/vit_large_patch14_clip_224.openai) and change the model_path in diffnext.rewards.reward_image.GenEvalScorer to your mmdetection path


the mmdetection format should be
```bash
mmdetection/
│
├── configs/
│   └── mask2former/
│       └── mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py
│
├── mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
│
├── vit_large_patch14_clip_224.openai/
│   ├── open_clip_config.json
│   ├── pytorch_model.bin   
│   └── ...
│
└── object_names.txt
```

#### 3. OCR 
Install the paddle-ocr and the model:
```
pip install paddlepaddle-gpu==2.6.2
pip install paddleocr==2.9.1
pip install python-Levenshtein

from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
```
change the ocr path in diffnext.rewards.reward_image.OCRScorer to your path

## 🥫 Data Preparation
<a id="datapreparation"></a>

GenEval 
```bash
# First
cd raw_rl_data/geneval
python cache.py

# Then Change the train_dataloader.params.dataset  in  ursa_1.7b_ibq512.yaml
```
The same way for PickScore and OCR.



## 🤖 Training
<a id="training"></a>

### 1. Single-node training
```bash
cd diffnext

accelerate launch --config_file accelerate_configs/4_nodes_deepspeed.yaml \
--machine_rank 0 --num_machines 1 --num_processes 8 \
scripts/train.py \
config="configs/geneval_grpo/ursa_1.7b_ibq512.yaml" \
experiment.name="ursa_geneval" \
experiment.output_dir="./experiments/ursa_geneval" 
```
**Note:** If you modify the batch size in the configuration, you must ensure that  
`training.batch_size = num_prompts * num_images // num_gpus // num_batches`.

### 2. Multi-node training
```bash
# Master node
sh scripts/geneval_grpo/main.sh

# Other nodes
sh scripts/geneval_grpo/main1.sh
sh scripts/geneval_grpo/main2.sh
sh scripts/geneval_grpo/main3.sh
```




## 🖋️ Evaluations
<a id="evaluation"></a>

### GenEval

#### 1. Sample prompt images
```bash
cd diffnext/evaluations/geneval

torchrun --nproc_per_node=8 sample.py \
--height 512 --width 512 \
--guidance_scale 1.0 --num_inference_steps 25 \
--ckpt /path/to/URSA-1.7B-IBQ512 \
--tdir /path/to/checkpoint-XXXX/transformer/diffusion_pytorch_model.bin \
--outdir ./output/URSA-1.7B-IBQ512 \
--distributed
```

#### 2. Evaluation
<IMAGE_FOLDER>=./output/URSA-1.7B-IBQ512

Please refer [GenEval](https://github.com/djghosh13/geneval?tab=readme-ov-file#evaluation) evaluation guide.

### PickScore

#### 1. Sample prompt images
```bash
cd diffnext/evaluations/pickscore

torchrun --nproc_per_node=8 sample.py \
--height 512 --width 512 \
--guidance_scale 1.0 --num_inference_steps 25 \
--ckpt /path/to/URSA-1.7B-IBQ512 \
--tdir /path/to/checkpoint-XXXX/transformer/diffusion_pytorch_model.bin \
--outdir ./output/URSA-1.7B-IBQ512 \
--distributed
```

#### 2. Evaluation
```bash
python evaluate.py \
--image_root ./output/URSA-1.7B-IBQ512 \
--out_file  ./output/URSA-1.7B-IBQ512/result.json
```


## 📖 Citation
If you find this repository useful, please consider giving a star ⭐ and citation 🦖:
```
@article{wang2026udmgrpo,
  title={UDM-GRPO: Stable and Efficient Group Relative Policy Optimization for Uniform Discrete Diffusion Models},
  author={Wang, Jiaqi and Deng, Haoge and Pan, Ting and Liu, Yang and Wang, Chengyuan and Zhang, Fan and Qi, Yonggang and Wang, Xinlong},
  journal={arXiv preprint arXiv:2604.18518},
  year={2026}
}
```
```
@article{deng2025ursa,
  title={Uniform Discrete Diffusion with Metric Path for Video Generation},
  author={Deng, Haoge and Pan, Ting and Zhang, Fan and Liu, Yang and Luo, Zhuoyan and Cui, Yufeng and Shen, Chunhua and Shan, Shiguang and Zhang, Zhaoxiang and Wang, Xinlong},
  journal={arXiv preprint arXiv:2510.24717},
  year={2025}
}
```
```
@article{deng2024nova,
  title={Autoregressive Video Generation without Vector Quantization},
  author={Deng, Haoge and Pan, Ting and Diao, Haiwen and Luo, Zhengxiong and Cui, Yufeng and Lu, Huchuan and Shan, Shiguang and Qi, Yonggang and Wang, Xinlong},
  journal={arXiv preprint arXiv:2412.14169},
  year={2024}
}
```

## 🤗 Acknowledgement

We thank the repositories: 
- [URSA](https://github.com/baaivision/URSA). 🐻URSA is the base model of UDM-GRPO.
- [NOVA](https://github.com/baaivision/NOVA). ✨NOVA is the predecessor of 🐻URSA.
- [CodeWithGPU](https://github.com/seetacloud/codewithgpu). CodeWithGPU library is the core of our data loading pipeline.

## License
Code and models are licensed under [Apache License 2.0](LICENSE).
