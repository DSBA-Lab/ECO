# Use NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.02-py3

# Set the working directory
WORKDIR /workspace


# Install system dependencies
RUN apt-get update && apt-get install -y wget curl bzip2 git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH /opt/conda/bin:$PATH


# Clone necessary repositories
RUN git clone https://github.com/baaivision/EVA.git && \
    git clone https://github.com/apple/ml-mobileclip.git

# Create Conda environments
RUN conda create -n blip2 python=3.8 -y && \
    conda create -n evaclip python=3.10 -y && \
    conda create -n clipenv python=3.10 -y

# Install BLIP2 dependencies
RUN conda run -n blip2 pip install --upgrade pip && \
    conda run -n blip2 pip install salesforce-lavis omegaconf

# Install EVA-CLIP dependencies
RUN conda run -n evaclip conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.6 -c pytorch -c conda-forge && \
    conda run -n evaclip pip install -r EVA/EVA-CLIP-18B/requirements.txt && \
    conda run -n evaclip pip install omegaconf pydantic==1.10.9 apex 
# Cleanup EVA clone
RUN rm -rf EVA

# Install CLIP environment dependencies
RUN conda run -n clipenv pip install ./ml-mobileclip && \
    conda run -n clipenv pip install omegaconf

# Cleanup ML MobileCLIP clone
RUN rm -rf ml-mobileclip

# Set the default command to bash
CMD [ "/bin/bash" ]
