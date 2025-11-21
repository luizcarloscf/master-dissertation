# dataset-creator  
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Linux-lightgrey.svg)

A set of Python scripts for dataset creation, including image capture, video generation, and gesture annotation tools.

## Installation

First, install the project dependencies. If you want to isolate them using a virtual environment, create one before installing. All direct dependencies are listed in `requirements.txt`. To install, run:

```bash
pip3 install -r requirements.txt
```

## Configuration

Edit the [options.json](options.json) file to set:
- `folder`: The directory where images will be saved.
- `broker_uri`: The AMQP broker URL used to receive images.

Ensure you have a **1000 Mi network connection**, which is required to receive **40 images per second** (4 cameras × 10 FPS). A simple way to check the incoming rate is by accessing the RabbitMQ management dashboard.

## Capturing Images

To record a person with ID `1` performing the gesture “Ask For Help”, execute:

```bash
python3 scripts/capture-images.py -p 1 -g 1
```

During execution:

- Press `s` to start recording.
- Press `s` again to stop recording.
- Press `q` to exit the program.

> :warning: You cannot exit the program while a recording is in progress.

Once the recording ends and the program closes, the images will be saved in the `folder` defined in [options.json](options.json).


## Generating Videos

To build videos from the captured images, run:

```bash
python3 scripts/make-videos.py
```


## Labeling Tool

The labeling tool allows you to annotate gesture segments in the videos. Because it loads the videos from all four cameras into memory simultaneously, it requires:

- **32 GB RAM** for videos up to **3 minutes**, or
- **16 GB RAM** for videos up to **1.5 minutes**.

Very long videos may not be supported. To start the labeling tool:

```bash
python3 scripts/label-videos.py
```

Navigation:

| Action            | Key | Description                     |
| ----------------- | --- | ------------------------------- |
| Next frame        | `k` | Moves forward by 1 frame        |
| Previous frame    | `j` | Moves backward by 1 frame       |
| Big forward step  | `l` | Moves forward by **10 frames**  |
| Big backward step | `h` | Moves backward by **10 frames** |


Creating Annotations:
| Action                 | Key | Description                                     |
| ---------------------- | --- | ----------------------------------------------- |
| Begin annotation       | `b` | Marks the start frame of a gesture              |
| End annotation         | `e` | Marks the end frame of a gesture                |
| Delete last annotation | `d` | Removes the gesture segment under the cursor |

Saving and Navigation:
| Action       | Key | Description                                 |
| ------------ | --- | ------------------------------------------- |
| Save labels  | `s` | Saves all annotations for the current video |
| Next video   | `n` | Loads the next video sequence               |
| Exit program | `q` | Closes the labeling tool                    |

> **Note**: A single video may contain multiple gesture intervals. Simply navigate to the desired segment and mark it using `b` and `e`.

For more details, see the [`keymap.json`](./keymap.json) file.

## Customizing the Dataset

If you want to customize the dataset to match your specific scenario, adding or removing gesture classes, simply edit the [`gestures.json`](./gestures.json) file to reflect your desired gesture vocabulary.