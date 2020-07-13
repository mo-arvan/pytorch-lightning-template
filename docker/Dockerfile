#https://github.com/PyTorchLightning/pytorch-lightning/blob/0.8.5/dockers/conda/Dockerfile
ARG CUDA_VERSION=10.2
FROM nvidia/cuda:${CUDA_VERSION}-base

# install versions
ARG PYTHON_VERSION=3.8
ARG LIGHTNING_VERSION=0.8.5


RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates

# add non-root user
RUN useradd --create-home --shell /bin/bash containeruser
USER containeruser
WORKDIR /home/containeruser

# install conda and python
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /home/containeruser/conda && \
    rm ~/miniconda.sh && \
    /home/containeruser/conda/bin/conda clean -ya && \
    /home/containeruser/conda/bin/conda install -y python=$PYTHON_VERSION 
     
# add conda to path
ENV PATH /home/containeruser/conda/bin:$PATH

# install dependencies

RUN pip install numpy && \
    pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html

RUN git clone https://github.com/PyTorchLightning/pytorch-lightning.git --single-branch --branch $LIGHTNING_VERSION && \
    pip install ./pytorch-lightning && \
    pip install -r pytorch-lightning/requirements/extra.txt && \
    rm -rf pytorch-lightning

RUN python -c "import pytorch_lightning as pl; print(pl.__version__)"

CMD ["/bin/bash"]

