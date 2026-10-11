FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb-cursor0 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libegl1 \
    libgl1 \
    libpipewire-0.3-0 \
    libxrandr2 \
    git \
    libglib2.0-0 \
    libfreetype6 \
    libfontconfig1 \
    fonts-dejavu-core \
    libpulse0 \
    libxcb-shape0 \
    && rm -rf /var/lib/apt/lists/*

ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
RUN python3 -m pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" 
RUN git clone --branch v2.6.3 \
    https://github.com/Dao-AILab/flash-attention.git \
    /tmp/flash-attention \
    && cd /tmp/flash-attention/csrc/fused_dense_lib \
    && python3 -m pip install . --no-build-isolation \
    && cd /tmp/flash-attention/csrc/layer_norm \
    && python3 -m pip install . --no-build-isolation \
    && rm -rf /tmp/flash-attention

# Install InternVideo2
RUN git clone https://github.com/loreleva/InternVideo.git /InternVideo
RUN python3 -m pip install /InternVideo/InternVideo2/multi_modality

# Download model weights
RUN python3 -m pip install --no-cache-dir huggingface_hub

# Bert
RUN hf download google-bert/bert-large-uncased \
    config.json \
    model.safetensors \
    special_tokens_map.json \
    tokenizer_config.json \
    vocab.txt \
    --local-dir /InternVideo/model_weights/bert-large-uncased

# InternVideo2
RUN hf download loreleva/InternVideo2-Stage2_1B-224p-f4 \
    InternVideo2-stage2_1b-224p-f4-vision_encoder.pt \
    InternVideo2-stage2_1b-224p-f4-text_encoder.pt \
    --local-dir /InternVideo/model_weights

# Install text_video_retrieval
COPY . /text_video_retrieval
RUN python3 -m pip install /text_video_retrieval

ENTRYPOINT [ "python3", "-m", "text_video_retrieval.gui.gui_app" ]