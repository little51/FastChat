# FastChat Fine-tuning

## 1、安装依赖环境

### 1.1 下载源码

```bash
git clone https://github.com/little51/FastChat
cd FastChat
```

### 1.2 安装FastChat运行环境

```bash
# 之前已经安装过可忽略
conda create -n fastchat python=3.10
conda activate fastchat 
python -m pip install --upgrade pip -i https://pypi.mirrors.ustc.edu.cn/simple --trusted-host=pypi.mirrors.ustc.edu.cn
pip3 install -e . -i https://pypi.mirrors.ustc.edu.cn/simple --trusted-host=pypi.mirrors.ustc.edu.cn
pip3 install --upgrade tokenizers -i https://pypi.mirrors.ustc.edu.cn/simple --trusted-host=pypi.mirrors.ustc.edu.cn
pip3 install protobuf==3.19.0 -i https://pypi.mirrors.ustc.edu.cn/simple --trusted-host=pypi.mirrors.ustc.edu.cn
# 验证torch是否安装正常
# 进行python
import torch
print("torch.cuda.is_available:",torch.cuda.is_available())
exit()
# 退出虚拟环境
conda deactivate
```

### 1.3  安装微调依赖环境

#### 1.3.1 install pkg-config

```bash
wget https://pkg-config.freedesktop.org/releases/pkg-config-0.29.2.tar.gz
tar -zxvf pkg-config-0.29.2.tar.gz
cd pkg-config-0.29.2
./configure --with-internal-glib
make -j4
make check
sudo make install
```

#### 1.3.2 install libicu

```bash
## centos (install by yum)
sudo yum install libicu-devel 
## ubuntu (install by apt)
sudo apt-get install libicu-dev
## centos/ubuntu (install from source)
wget https://mirrors.aliyun.com/blfs/conglomeration/icu/icu4c-73_1-src.tgz
tar xf icu4c-73_1-src.tgz
cd icu/source
./configure
make
make check
sudo make install
```

#### 1.3.3 install packages

```bash
conda activate fastchat
pip3 install -r ./fine-tuning/requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple --trusted-host=pypi.mirrors.ustc.edu.cn
```

## 2、准备Llama模型

### 2.1 下载Llama模型

```bash
export GIT_TRACE=1
export GIT_CURL_VERBOSE=1
pip3 install git+https://github.com/juncongmoo/pyllama -i https://pypi.mirrors.ustc.edu.cn/simple --trusted-host=pypi.mirrors.ustc.edu.cn
python -m llama.download --model_size 7B
```

### 2.2 转换模型为huggingface格式

```bash
CUDA_VISIBLE_DEVICES=1 python3 ./fine-tuning/convert_llama_weights_to_hf.py --input_dir ./pyllama_data --model_size 7B --output_dir ./pyllama_data/output/7B
```

## 3、整理语料

### 3.1 语料下载

下载52k的ShareGPT：https://huggingface.co/datasets/RyokoAI/ShareGPT52K

其他语料参见：https://github.com/Zjh-819/LLMDataHub

下载的sg_90k_part1.json和sg_90k_part2.json放到fine-tuning/data/sharegpt52k

### 3.2 合并语料文件

```bash
python3 -m fastchat.data.merge --in ./fine-tuning/data/sharegpt52k/sg_90k_part1.json ./fine-tuning/data/sharegpt52k/sg_90k_part2.json --out  ./fine-tuning/data/sharegpt52k/sg_90k.json
```

### 3.3 html转markdown

```bash
python3 -m fastchat.data.clean_sharegpt --in ./fine-tuning/data/sharegpt52k/sg_90k.json --out ./fine-tuning/data/sharegpt52k/sharegpt_clean.json
```

### 3.4 去掉一些用不到的语言（可选）

```bash
python3 -m fastchat.data.optional_clean --in ./fine-tuning/data/sharegpt52k/sharegpt_clean.json --out ./fine-tuning/data/sharegpt52k/sharegpt_clean_1.json --skip-lang SOME_LANGUAGE_CODE
其中SOME_LANGUAGE_CODE的取值举例如下：
en - 英语
es - 西班牙语
fr - 法语
de - 德语
it - 意大利语
ja - 日语
ko - 朝鲜语
zh - 中文
ar - 阿拉伯语
ru - 俄语
pt - 葡萄牙语
nl - 荷兰语
```

### 3.5 将长会话切分成短对话

```shell
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.data.split_long_conversation --in ./fine-tuning/data/sharegpt52k/sharegpt_clean.json --out ./fine-tuning/data/sharegpt52k/sharegpt_clean_split.json --model-name ./pyllama_data/output/7B
```

## 4、微调

### 4.1 常规方法

```bash
CUDA_VISIBLE_DEVICES=1 \
torchrun --nproc_per_node=1 --master_port=20001 \
fastchat/train/train_mem.py \
    --model_name_or_path ./pyllama_data/output/7B  \
    --data_path ./fine-tuning/data/sharegpt52k/sharegpt_clean_split.json \
    --fp16 True \
    --output_dir ./fine-tuning/output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "no_shard" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True
```

### 4.2 lora方法

```bash
CUDA_VISIBLE_DEVICES=1 \
deepspeed fastchat/train/train_lora.py \
    --deepspeed ./fine-tuning/deepspeed.config \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --model_name_or_path ./pyllama_data/output/7B \
    --data_path ./fine-tuning/data/sharegpt52k/sharegpt_clean_split.json \
    --fp16 True \
    --output_dir ./fine-tuning/output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048
```

### 4.3 T5模型

```bash
mkdir google
cd google
git clone https://huggingface.co/google/flan-t5-xl
CUDA_VISIBLE_DEVICES=1 \
torchrun --nproc_per_node=1 --master_port=20001 \
	fastchat/train/train_flant5.py \
    --model_name_or_path ./opt-125m \
    --data_path ./fine-tuning/data/sharegpt52k/sharegpt_clean_split.json \
    --output_dir ./fine-tuning/output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap T5Block \
    --model_max_length 2048 \
    --preprocessed_path ./preprocessed_data/processed.json \
    --gradient_checkpointing True
```



## 参考

https://github.com/lm-sys/FastChat/pull/177

https://github.com/h2oai/h2ogpt/pull/86

https://github.com/lm-sys/FastChat/issues/608

https://zhuanlan.zhihu.com/p/618073170

https://zhuanlan.zhihu.com/p/617221484



