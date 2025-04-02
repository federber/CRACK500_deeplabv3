# Используем базовый образ с поддержкой Python и CUDA
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Устанавливаем необходимые зависимости
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libxrender1 \
    libxext6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем зависимости Python
WORKDIR /app

COPY deeplabv3_crack.py /app/
COPY utils.py /app/

RUN pip3 install --no-cache-dir torch opencv-python numpy pandas scikit-learn argparse segmentation_models_pytorch torchmetrics


# Указываем рабочую директорию
WORKDIR /app

# Запуск скрипта
CMD ["python3", "deeplabv3_crack.py"]