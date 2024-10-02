# Face Comparison Tool

This tool allows you to compare two photos of faces and determine their similarity.

Created with the help of [Aider](https://aider.chat).

## Installation

1. Ensure you have Python installed on your system.
2. Install the required dependencies using `uv`:
   ```sh
   uv sync
   ```

## Usage

### Compare

```shell
uv run compare.py compare <path_to_image1> <path_to_image2>
```

This command will output the similarity score between the two images. A higher score indicates greater similarity.

### Debug

To show the bounding boxes of the detected faces in the images, use:
```shell
uv run compare.py debug <path_to_image1> <path_to_image2>
```

