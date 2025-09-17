# eDNA Backend API

This repository contains the backend API for the eDNA classification project. It uses **FastAPI** to serve a **PyTorch** deep learning model that predicts the phylum of an organism based on a raw DNA sequence.

## Getting Started

Follow these instructions to get a local copy of the project up and running for development and testing.

### Prerequisites

Make sure you have **Python 3.8+** installed on your system.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/ssrithwik007/eDNA-backend.git
    cd eDNA-backend
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    # Create the virtual environment
    python -m venv venv

    # Activate it:
    # On Windows (PowerShell)
    .\venv\Scripts\activate

    # On macOS / Linux
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    This command reads the `requirements.txt` file and installs all the necessary libraries for the project.

    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

Once the installation is complete, you can run the API server.

From the project's root directory (`backend`), run the following command to start the **Uvicorn** server:

```bash
uvicorn app.main:app --reload
```

The `--reload` flag makes the server restart automatically when you make changes to the code.

The API will now be running and available at **`http://127.0.0.1:8000`**. 🚀

## API Usage

You can interact with the API in two primary ways:

1.  **Interactive Docs (Recommended):**
    Navigate to **`http://127.0.0.1:8000/docs`** in your browser to see the automatically generated Swagger UI documentation. You can test the endpoints directly from this page.

2.  **Using `curl`:**
    You can send a POST request to the `/predict` endpoint from your terminal.

    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/predict' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "sequence": "GTCGATCGGCTAGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCA"
    }'
    ```
