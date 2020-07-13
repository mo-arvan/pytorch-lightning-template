# pytorch-lightning-template

## Docker
Building a docker image with torch nightly.

```bash
cd /path/to/workspace/pytorch-lightning-template
docker build -t pl-template --build-arg PYTHON_VERSION=3.8 --build-arg CUDA_VERSION=10.2 --build-arg LIGHTNING_VERSION=0.8.5 -f docker/Dockerfile .
```

## Run

```bash
cd /path/to/workspace/pytorch-lightning-template
docker run --gpus all -it --rm --ipc=host -v $PWD:/home/containeruser/workspace pl-template  python3 workspace/gpu_template.py
```

## Credits
- [Pytorch Lightning Example](https://github.com/PyTorchLightning/pytorch-lightning/blob/0.8.5/pl_examples/)
- [Pytorch Lightning Dockerfile](https://github.com/PyTorchLightning/pytorch-lightning/blob/0.8.5/dockers/conda/Dockerfile)
