# When Open-Vocabulary Visual Question Answering Meets Causal Adapter: Benchmark and Approach(accepted to AAAI2025)

## Introduction

### 1. Motivation and VQACL Setting

While the VQA community has achieved substantial progress, the benchmarks employed thus far have predominantly adhered to the closed-set setting. As depicted in following figure (a), these benchmarks rely on a predefined set of candidate answers (e.g., baseball, skyblue), restricting models to select from this fixed list and impeding their ability to handle unseen concepts (e.g., badminton, citizen). This constraint diminishes model effectiveness in open-world scenarios, where answer distributions are highly varied and dynamic. 

<div align="center">
  <img src="https://carmacchiato-blog.oss-cn-beijing.aliyuncs.com/img/blog/OVVQA202412181859408.png" alt="image-20241218185910329" width="400px" />
</div>

To overcome these limitations, we propose a new evaluation benchmark for VQA, termed Open-Vocabulary Visual Question Answering (OVVQA). OVVQA is designed to better align with open-world conditions and to provide a more accurate assessment of a model’s multimodal reasoning abilities.

### 2. Method

We propose a novel Causal Adapter to efficiently transfer causal knowledge from the pretrained model to OVVQA, which is designed in a plug-and-play manner for ease of integration. Our framework enables the model to identify the causal relationship between inputs (e.g., visual content) and outputs (e.g., answers), thereby mitigating biases caused by distribution discrepancies between base and novel classes in OVVQA. We begin by formulating OVVQA as a causal graph and analyzing how existing methods establish spurious associations between visual content and answers. Spurious effects are then eliminated through a causal intervention based on the front-door adjustment principle, which is incorporated into the adapter and does not require the assumption of any observed confounder. This makes the proposed Causal Adapter applicable across various domains where the adapter resides. Additionally, we introduce an adaptive transfer loss to improve the fine-tuning process, effectively leveraging knowledge from pretext tasks to enhance performance across base and novel classes.

![image-20241218190431740](https://carmacchiato-blog.oss-cn-beijing.aliyuncs.com/img/blog/OVVQA202412181904766.png)

### 3. Experiment

The table presents the experimental results on our reconstituted OVVQA datasets: OV-VQAv2, OV-GQA, and OV-OKVQA. We report performance across several aspects, including results for base and novel classes, arithmetic mean (Avg), and harmonic mean (H). For OVVQAv2, we further subdivide the novel class into three subcategories: Yes/No (Y/N), Number (Num.), and Other.

<div align="center">
  <img src="https://carmacchiato-blog.oss-cn-beijing.aliyuncs.com/img/blog/OVVQA202412181907742.png" alt="image-20241218190745712" style="zoom:80%;" />
</div>

## Setup

```bash
# Create python environment (optional)
conda create -n causala python=3.8
source activate causala

# Install python dependencies
pip install -r requirements.txt

# Download T5/BART backbone checkpoint
python download_backbones.py
```

## Code structure
```bash
# Store images, features, and annotations
./datasets
    COCO/
        images/
        featuers/
    VG/
        images/
        features/
    GQA/
        images/
        features/
    OVVQAv2/
    ...

# Run feature extraction
./feature_extraction

# Train VL-T5
./VL-T5/
    src/
        modeling_t5.py modeling_bart.py                       <= VL-T5/VL-BART model classes
        modeling_adapter.py                                   <= Causal_Adapter
        pretrain.py, pretrain_data.py, pretrain_model.py      <= pretraining
        vqa.py, vqa_data.py, vqa_model.py, gqa.py ...         <= fine-tuning on downstream 
        param.py                                              <= (argparse) configuration
        tokenization.py                                       <= custom tokenizer
        utils.py, dist_utils.py                               <= utility functions
    snap/                                                     <= store weight checkpoints
    scripts/                                                  <= bash scripts for pretraining and finetuning
```

## Dataset Preparation / Model checkpoint

- Download the OVVQAv2 partition from [Google Drive](https://drive.google.com/drive/u/0/folders/1Due4eyWZjVZhtfTiibAeJmK_971sdVhI) and put it into datasets/OVVQAv2.
- Download the OVGQA partition from [Google Drive](https://drive.google.com/drive/u/0/folders/1ppEe1ta24UWlpbqNcD01GSuIPzU57EN3) and put it into datasets/OVGQA.
- Download the OVOKVQA partition from [Google Drive](https://drive.google.com/drive/u/0/folders/1D5so31vmcXzfyB4yQQXhxRnM0k-a0M0U) and put it into datasets/OVOKVQA.
- Download `datasets/COCO` from [Google Drive](https://drive.google.com/drive/folders/1MBBhlkP83VMKS2Qe0SmFfzkHhMpIG5wf?usp=sharing)
- Download `datasets/VG` from [Google Drive](https://drive.google.com/drive/folders/1MBBhlkP83VMKS2Qe0SmFfzkHhMpIG5wf?usp=sharing)
- Download model checkpoints from [Google Drive](https://drive.google.com/drive/u/0/folders/1WdW3KqHdlJQN7BQmK1jsOyXCy4QfbF5f)

## OVVQA Tasks
```bash
# Training with 1 gpu for VQA v2
cd VL-T5/
bash scripts/VQA_VLT5.sh       # Standard training based VL-T5
bash scripts/VQA_VLBART.sh       # Standard training based VL-BART

# Training with 1 gpu for GQA
cd VL-T5/
bash scripts/GQA_VLT5.sh       # Standard training based VL-T5
bash scripts/GQA_VLBART.sh       # Standard training based VL-BART

# Training with 1 gpu for OKVQA
cd VL-T5/
bash scripts/OKVQA_VLT5.sh       # Standard training based VL-T5
bash scripts/OKVQA_VLBART.sh       # Standard training based VL-BART
```

## Acknowledgement

Our model is based on the official [VL-T5](https://github.com/j-min/VL-T5) repository, we thank the authors to release their code. If you use the related part, please cite the corresponding paper commented in the code.
