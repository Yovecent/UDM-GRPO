from setuptools import setup, find_packages

setup(
    name="udm-grpo",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        "transformers==4.51.1",
        "accelerate==1.4.0",
        "diffusers==0.33.1", 
        
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scipy==1.15.2",
        "scikit-learn==1.6.1",
        "scikit-image==0.25.2",
        
        "albumentations==1.4.10",  
        "opencv-python==4.11.0.86",
        "pillow==10.4.0",
        
        "tqdm==4.67.1",
        "codewithgpu==0.2.8",
        "protobuf==3.20.3",
        "wandb==0.15.12",
        "pydantic==2.10.6",  
        "requests",
        "matplotlib==3.10.0",
        "setuptools==78.1.1",
        "deepspeed==0.16.4",  
        "peft==0.10.0",       
        "bitsandbytes==0.45.3",
        "omegaconf==2.3.0",
        "aiohttp==3.11.13",
        "fastapi==0.115.11", 
        "uvicorn==0.34.0",
        
        "huggingface-hub==0.34.4",  
        "datasets==3.3.2",
        "tokenizers==0.21.4",
        
        "einops==0.8.1",
        "nvidia-ml-py==12.570.86",
        "absl-py",
        "ml_collections",
        "sentencepiece",

    ],
    extras_require={
        "dev": [
            "ipython==8.34.0",
            "black==24.2.0",
            "pytest==8.2.0"
        ]
    }
)