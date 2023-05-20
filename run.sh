wget https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh
chmod a+x Miniconda3-py37_23.1.0-1-Linux-x86_64.sh
./Miniconda3-py37_23.1.0-1-Linux-x86_64.sh

conda env create -f environment.yml
conda activate minigpt4
sudo apt-get update
sudo apt -y install git wget git-lfs
git lfs install
mkdir 13B
mkdir llama-13b-hf

git lfs install
git clone https://huggingface.co/lmsys/vicuna-13b-delta-v0  # more powerful, need at least 24G gpu memory
cd vicuna-13b-delta-v0/
git lfs pull
wget https://agi.gpt4.org/llama/LLaMA/tokenizer.model -O ./tokenizer.model
wget https://agi.gpt4.org/llama/LLaMA/tokenizer_checklist.chk -O ./tokenizer_checklist.chk
wget https://agi.gpt4.org/llama/LLaMA/13B/consolidated.00.pth -O ./13B/consolidated.00.pth
wget https://agi.gpt4.org/llama/LLaMA/13B/consolidated.01.pth -O ./13B/consolidated.01.pth
wget https://agi.gpt4.org/llama/LLaMA/13B/params.json -O ./13B/params.json
wget https://agi.gpt4.org/llama/LLaMA/13B/checklist.chk -O ./13B/checklist.chk
git clone https://github.com/huggingface/transformers.git
/home/ubuntu/miniconda3/envs/minigpt4/bin/pip install tokenizers --upgrade
conda uninstall tokenizers
python -m pip install protobuf==3.19.2
python transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir ./ --model_size 13B --output_dir ./llama-13b-hf/

python -m pip install git+https://github.com/lm-sys/FastChat.git@v0.1.10
python -m fastchat.model.apply_delta --base ./llama-13b-hf/  --target ./13B  --delta ./vicuna-13b-delta-v0/


