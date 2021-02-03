# DeBERTa: Decoding-enhanced BERT with Disentangled Attention, 解码增强的解码注意力BERT

论文地址： [ **DeBERTa**: **D**ecoding-**e**nhanced **BERT** with Disentangled **A**ttention ](https://arxiv.org/abs/2006.03654)

## News
### 12/29/2020
使用DeBERTa 1.5B模型，我们在SuperGLUE排行榜上超越了T5 11B模型和人类表现。 代码和模型将很快发布。 请查看我们的论文以获取更多详细信息。

### 06/13/2020
我们发布了预训练的模型，源代码和微调脚本，以重现本文中的一些实验结果。 
您可以按照类似的脚本将DeBERTa应用于您自己的实验或应用程序。 下一步将发布预训练脚本。 


## Introduction to DeBERTa 
DeBERTa(注意力解耦的增强解码的BERT)使用两种新颖的技术改进了BERT和RoBERTa模型。
首先是解耦的的注意力机制，其中每个单词分别使用两个编码其内容和位置的向量表示，
单词间的注意力权重使用其内容和相对位置的解耦的矩阵来计算。 
其次，增强的mask解码器用于替换输出softmax层，以预测用于模型预训练的mask token。 
我们证明了这两种技术显着提高了模型预训练的效率和下游任务的性能。

# Pre-trained Models
我们预训练的模型打包成压缩文件。 您可以从我们的[releasements](https://github.com/microsoft/DeBERTa/releases)下载它们，或通过以下链接下载单个模型：
- [Large](https://github.com/microsoft/DeBERTa/releases/download/v0.1/large.zip): the pre-trained Large model
- [Base](https://github.com/microsoft/DeBERTa/releases/download/v0.1.8/base.zip) : the pre-trained Base model
- [Large MNLI](https://github.com/microsoft/DeBERTa/releases/download/v0.1/large_mnli.zip): Large model fine-tuned with MNLI task
- [Base MNLI](https://github.com/microsoft/DeBERTa/releases/download/v0.1/base_mnli.zip): Base model fine-tuned with MNLI task


# Try the code

详细文档 [documentation](https://deberta.readthedocs.io/en/latest/)

## Requirements
- Linux system, e.g. Ubuntu 18.04LTS
- CUDA 10.0
- pytorch 1.3.0
- python 3.6
- bash shell 4.0
- curl
- docker (optional)
- nvidia-docker2 (optional)

可以使用以下几种方案使用模型
### Use docker
提议使用Docker运行代码，因为我们已经在docker[bagai/deberta](https://hub.docker.com/r/bagai/deberta)中建立了每个依赖关系，
您可以按照[docker official site](https://docs.docker.com/engine/install/ubuntu/)将docker安装到您的机器上。 

要与docker一起运行，请确保您的系统满足上述列表中的要求。 以下是尝试GLUE实验的步骤：拉出代码，  运行 `./run_docker.sh` 
，然后您可以在下面运行bash命令 `/DeBERTa/experiments/glue/`

### Use pip
拉取代码并在代码的根目录中运行`pip3 install -r requirements.txt`，然后进入代码的`experiments/glue/`文件夹并尝试在该文件夹下的bash命令进行glue实验。 

### Install as a pip package
`pip install deberta`

#### Use DeBERTa in existing code
``` Python

# 要将DeBERTa应用于现有代码，您需要对代码进行两项更改， 
# 1. 更改模型以使用DeBERTa作为编码器 
from DeBERTa import deberta
import torch
class MyModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    # Your existing model code
    self.bert = deberta.DeBERTa(pre_trained='base') # Or 'large' or 'base_mnli' or 'large_mnli'
    # do inilization as before
    # 在构造函数的末尾应用DeBERTa的预训练模型, 就是加载预训练模型的参数
    self.bert.apply_state() 
    #
  def forward(self, input_ids):
    # DeBERTa前向输入输入为 
    # `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] with the word token indices in the vocabulary
    # `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token types indices selected in [0, 1]. 
    #    Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
    # `attention_mask`: an optional parameter for input mask or attention mask. 
    #   - If it's an input mask, then it will be torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1]. 
    #      如果输入序列长度小于当前批次中的最大输入序列长度，则使用此mask。 
    #      当一批具有不同长度的句子时，用mask的注意力。 
    #   - If it's an attention mask then if will be torch.LongTensor of shape [batch_size, sequence_length, sequence_length]. 
    #      In this case, it's a mask indicate which tokens in the sequence should be attended by other tokens in the sequence. 
    # `output_all_encoded_layers`: whether to output results of all encoder layers, default, True
    encoding = self.bert(input_ids)[-1]

# 2. 更改tokenizer, 使用DeBERta内置的tokenizer 
from DeBERTa import deberta
tokenizer = deberta.GPT2Tokenizer()
# 我们应用与BERT相同的特殊token模式 , e.g. [CLS], [SEP], [MASK]
max_seq_len = 512
tokens = tokenizer.tokenize('Examples input text of DeBERTa')
#截断长序列
tokens = tokens[:max_seq_len -2]
#将特殊token添加到“token”中 
tokens = ['[CLS]'] + tokens + ['[SEP]']
input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_mask = [1]*len(input_ids)
# padding
paddings = max_seq_len-len(input_ids)
input_ids = input_ids + [0]*paddings
input_mask = input_mask + [0]*paddings
features = {
'input_ids': torch.tensor(input_ids, dtype=torch.int),
'input_mask': torch.tensor(input_mask, dtype=torch.int)
}

```

#### Run DeBERTa experiments from command line
For glue tasks, 
1. Get the data
``` bash
cache_dir=/tmp/DeBERTa/
#github链接已失效
curl -J -L https://raw.githubusercontent.com/nyu-mll/jiant/master/scripts/download_glue_data.py | python3 - --data_dir $cache_dir/glue_tasks
python3 utils/download_glue_data.py --data_dir cache_dir
```
2. Run task

``` bash
task=STS-B 
OUTPUT=/tmp/DeBERTa/exps/$task
export OMP_NUM_THREADS=1
python3 -m DeBERTa.apps.train --task_name $task --do_train  \
  --data_dir $cache_dir/glue_tasks/$task \
  --eval_batch_size 128 \
  --predict_batch_size 128 \
  --output_dir $OUTPUT \
  --scale_steps 250 \
  --loss_scale 16384 \
  --accumulative_update 1 \  
  --num_train_epochs 6 \
  --warmup 100 \
  --learning_rate 2e-5 \
  --train_batch_size 32 \
  --max_seq_len 128
```

#使用 python的 module运行
```buildoutcfg
cd /Users/admin/git/DeBERTa

python -m DeBERTa.apps.train \
--task_name STS-B --do_train --do_eval --do_predict --pre_trained output/base/pytorch.model.bin --model_config output/base/model_config.json --data_dir data/glue_tasks/STS-B --eval_batch_size 128 --bpe_vocab_file output/base/bpe_encoder.bin --predict_batch_size 128 --output_dir output/SST-B --scale_steps 250 --loss_scale 16384 --accumulative_update 1 --num_train_epochs 6 --warmup 100 --learning_rate 2e-5 --train_batch_size 32 --max_seq_len 128
```


## Important Notes
1. 要在多个GPU上运行我们的代码，在启动我们的训练代码之前，您必须设置环境变量`OMP_NUM_THREADS = 1` 
2. 默认情况下，我们将在`$HOME/.~DeBERTa`中缓存经过预训练的模型和tokenizer，如果下载意外失败，则可能需要清除它。 


## Experiments
我们的微调实验是在带有8x32 V100 GPU卡的DGX-2节点的上进行的，结果可能因GPU模型，驱动程序，使用FP16或FP32的CUDA SDK版本不同以及随机种子而异。 
我们在这里根据具有不同随机种子的多次运行报告我们的数字。 以下是Large模型的结果：

|Task	 |Command	|Results	|Running Time(8x32G V100 GPUs)|
|--------|---------------|---------------|-------------------------|
|MNLI xlarge|	`experiments/glue/mnli_xlarge.sh`|	91.5/91.4 +/-0.1|	2.5h|
|MNLI large|	`experiments/glue/mnli_large.sh`|	91.2/91.0 +/-0.1|	2.5h|
|QQP large|	`experiments/glue/qqp_large.sh`|	92.3 +/-0.1|		6h|
|QNLI large|	`experiments/glue/qnli_large.sh`|	95.3 +/-0.2|		2h|
|MRPC large|	`experiments/glue/mrpc_large.sh`|	93.4 +/-0.5|		0.5h|
|RTE large|	`experiments/glue/rte_large.sh`|	87.7 +/-1.0|		0.5h|
|SST-2 large|	`experiments/glue/sst2_large.sh`|	96.7 +/-0.3|		1h|
|STS-b large|	`experiments/glue/Stsb_large.sh`|	92.5 +/-0.3|		0.5h|
|CoLA large|	`experiments/glue/cola_large.sh`|	70.5 +/-1.0|		0.5h|

Base模型的结果

|Task	 |Command	|Results	|Running Time(8x32G V100 GPUs)|
|--------|---------------|---------------|-------------------------|
|MNLI base|	`experiments/glue/mnli_base.sh`|	88.8/88.5 +/-0.2|	1.5h|

## Contacts

Pengcheng He(penhe@microsoft.com), Xiaodong Liu(xiaodl@microsoft.com), Jianfeng Gao(jfgao@microsoft.com), Weizhu Chen(wzchen@microsoft.com)

# Citation
```
@misc{he2020deberta,
    title={DeBERTa: Decoding-enhanced BERT with Disentangled Attention},
    author={Pengcheng He and Xiaodong Liu and Jianfeng Gao and Weizhu Chen},
    year={2020},
    eprint={2006.03654},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

