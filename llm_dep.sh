export CUDA_VISIBLE_DEVICES=0,3,4,5
export HF_ENDPOINT=https://huggingface.com

python -m vllm.entrypoints.openai.api_server \
    --model /home/guest/models/llama-8B-instruct \
    --load-format safetensors \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --port 8888 \
    --tensor-parallel-size 4 \