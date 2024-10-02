from typing import Any
from insightface.app import FaceAnalysis
import cv2
import typer
import numpy as np

app = typer.Typer()


def _load_faces(
    photo1_path: str,
    photo2_path: str,
) -> tuple[Any, Any]:
    img1 = cv2.imread(photo1_path)
    img2 = cv2.imread(photo2_path)

    # Initialize face analysis application
    app = FaceAnalysis()
    app.prepare(ctx_id=-1)  # use CPU (for macOS)

    # Load images
    face1 = app.get(img1)
    face2 = app.get(img2)

    if not len(face1) or not len(face2):
        typer.echo("No faces detected in one or both images.")
        return

    return img1, img2, face1, face2


@app.command()
def debug(
    photo1_path: str = typer.Argument(..., help="File 1 path"),
    photo2_path: str = typer.Argument(..., help="File 2 path"),
):
    img1, img2, face1, face2 = _load_faces(photo1_path, photo2_path)

    def draw_bounding_boxes(image, faces):
        for face in faces:
            # Get the bounding box coordinates (x, y, width, height)
            x1, y1, x2, y2 = [int(coord) for coord in face.bbox]
            # Draw the rectangle around the face
            cv2.rectangle(
                image, (x1, y1), (x2, y2), (0, 255, 0), 2
            )  # Green rectangle with thickness 2
        return image

    # Draw bounding boxes on the images
    img1_with_boxes = draw_bounding_boxes(img1, face1)
    img2_with_boxes = draw_bounding_boxes(img2, face2)

    # Save the images with the bounding boxes
    cv2.imwrite(
        ".".join(photo1_path.split(".")[:-1]) + "_with_boxes.jpg", img1_with_boxes
    )
    cv2.imwrite(
        ".".join(photo2_path.split(".")[:-1]) + "_with_boxes.jpg", img2_with_boxes
    )

    # Optionally, display the images with bounding boxes
    cv2.imshow("Image 1 with Bounding Boxes", img1_with_boxes)
    cv2.imshow("Image 2 with Bounding Boxes", img2_with_boxes)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()  # Close the image windows


@app.command()
def compare(
    photo1_path: str = typer.Argument(..., help="File 1 path"),
    photo2_path: str = typer.Argument(..., help="File 2 path"),
):
    _, __, face1, face2 = _load_faces(photo1_path, photo2_path)

    # Get face embeddings
    embedding1 = face1[0].embedding
    embedding2 = face2[0].embedding

    # Calculate similarity (cosine similarity or other metric)
    similarity = np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )
    print("Similarity score:", similarity)


if __name__ == "__main__":
    # run typer app
    app()
