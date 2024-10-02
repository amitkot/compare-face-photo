import insightface
from insightface.app import FaceAnalysis
import typer
import numpy as np

app = typer.Typer()

@app.command()
def compare(
    photo1_path: str = typer.Argument(..., help="File 1 path"),
    photo2_path: str = typer.Argument(..., help="File 2 path"),
):
    # Initialize face analysis application
    app = FaceAnalysis()
    app.prepare(ctx_id=-1)  # use CPU (for macOS)

    # Load images
    img1 = insightface.utils.face_align(photo1_path)
    img2 = insightface.utils.face_align(photo2_path)

    # Get face embeddings
    embedding1 = app.get(img1)[0].embedding
    embedding2 = app.get(img2)[0].embedding

    # Calculate similarity (cosine similarity or other metric)
    similarity = np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )
    print("Similarity score:", similarity)

if __name__ == "__main__":
    # run typer app
    app()
