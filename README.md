# Gauge Read

Gauge Read is a comprehensive solution for reading pointer meters in complex environments. It includes a main meter reading model, a Spatial Transformer Network (STN) for correcting perspective distortion, and a YOLO-based meter detection model. The project provides a unified command line interface for training, validation, inference, and launching web and desktop interfaces.

## Installation

- Install uv. Use the following command to install uv, or use `pip install uv` if you have pip installed.

    ```bash
    # Windows
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    # Linux & MacOS
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

- Clone the repository and install the dependencies.

    ```bash
    git clone https://github.com/SUC-DriverOld/gauge-read.git
    cd gauge-read
    uv sync
    ```

- Download the pretrained models from the [release page](https://github.com/SUC-DriverOld/gauge-read/releases/tag/weights) and place them in the `pretrain` directory (mkdir if not exists) as follows:

    ```
    gauge-read/pretrain
        ├─meter
        │   ├── meter_convnext_tiny_ep100.pth
        │   └── meter_convnext_tiny_ep100.yaml (Optional)
        ├─stn
        │   └── stn_ep35_loss0.0551.pth
        └─yolo
            └── yolo26_best.pt
    ```

- Use `gaugeread` command to run the project. Use `gaugeread -h` to see the available subcommands and options.

    ```bash
    usage: gaugeread <subcommand> [options]

    Gauge Read unified command line entrypoint

    positional arguments:
    {api,train,train-stn,valid,infer,web,webui,gui}

    Available subcommands:
        api       Launch the FastAPI service
        train     Train the main meter reading model
        train-stn Train the STN correction model
        valid     Run validation on a labeled dataset
        infer     Run single-image CLI inference
        web       Launch the native WebUI
        gui       Launch the desktop GUI wrapper for the WebUI

    options:
    -h, --help            show this help message and exit
    ```

    To launch the WebUI, run:

    ```bash
    gaugeread web
    ```

## Training and Validation

You can train the main meter reading model and the STN correction model using the `train` and `train-stn` subcommands. You can download the meter reading model dataset from the [release page](https://github.com/SUC-DriverOld/gauge-read/releases/tag/weights), unzip it, and place it in the `datas` directory. STN model use generated data, so you can directly run the training command without additional data preparation. To train with your own dataset, you need to modify the `gauge_read/configs/config.yaml` file accordingly.

- Train the main meter reading model (use `gaugeread train -h` to see more options)

    ```bash
    gaugeread train -c config.yaml
    ```

- Train the STN correction model (use `gaugeread train-stn -h` to see more options)

    ```bash
    gaugeread train-stn -c config.yaml
    ```

- You can run validation on a labeled dataset using the `valid` subcommand. You need to prepare a labeled dataset in the same format as the training data and specify its path in the `config.yaml` file.

    ```bash
    gaugeread valid -c config.yaml
    ```

## Inference

We provide several options for running inference on images, including single-image CLI inference, a native HTML/CSS/JS Web UI, and a desktop GUI wrapper. You can use the `infer`, `web`, and `gui` subcommands to launch these interfaces.

We also provide api service for inference. You can launch the FastAPI service using the `api` subcommand, and then send POST requests to the `/infer` endpoint with an image file to get the meter reading results.

## Acknowledgments

Thank you for the contribution of the following paper and its code to this project.

```
@misc{shu2023read,
    title={Read Pointer Meters in complex environments based on a Human-like Alignment and Recognition Algorithm}, 
    author={Yan Shu and Shaohui Liu and Honglei Xu and Feng Jiang},
    year={2023},
    eprint={2302.14323},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
