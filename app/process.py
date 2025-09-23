import base64
import io
from typing import List
import numpy as np
import pandas as pd
import torch
import os
import umap
from Bio import SeqIO
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.spatial.distance as distance

SEQ_LENGTH = 300
CLASSES = ['Cnidaria', 'Arthropoda', 'Porifera', 'Echinodermata']

def one_hot_encode(seq: str):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    encoded = np.zeros((4, len(seq)), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        if base in mapping and mapping[base] != 4:
            encoded[mapping[base], i] = 1.0
    return encoded

def compute_prediction(model, input_tensor, label_encoder):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
        probabilities = torch.softmax(output, dim=1)
        confidence = probabilities[0][predicted_class].item()*100

    predicted_label = label_encoder.inverse_transform(predicted_class.cpu().numpy())

    return predicted_label[0], confidence

def get_embeddings(extractor, input_tensor):
    extractor.eval()
    with torch.no_grad():
        embedding = extractor(input_tensor)

    return embedding.cpu().numpy().squeeze()

def extract_fasta(input_file):
    ids = []
    descriptions = []
    sequences = []
    lengths = []

    # Use SeqIO.parse() to iterate over each sequence record
    # The input_file can be a file handle from a local file or a web upload
    for record in SeqIO.parse(input_file, "fasta"):
        # record is a SeqRecord object
        
        # Append data to the respective lists
        ids.append(record.id)
        descriptions.append(record.description)
        # It's good practice to convert the Seq object to a string
        sequences.append(str(record.seq))
        lengths.append(len(record.seq))

    # Create a dictionary from the lists
    data = {
        'ID': ids,
        'Description': descriptions,
        'Sequence': sequences,
        'Length (bp)': lengths
    }

    # Create the Pandas DataFrame
    df = pd.DataFrame(data)
    
    # Handle cases where no data was parsed
    if df.empty:
        return df

    # Use a regular expression to extract the organism name (e.g., "Giardia lamblia")
    # This pattern captures the text between the first word (the ID) and the phrase "18S".
    # We use .fillna('') to handle cases where the pattern might not match.
    df['label'] = df['Description'].str.extract(r'^\S+\s(.*?)\s18S.*$')[0].fillna('')

    return df    

def process_sequences(sequences: list[str], feature_extractor, device: str = "cpu"):
    feature_extractor.to(device)
    feature_extractor.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 4, SEQ_LENGTH)
        emb = feature_extractor(dummy)
    padded_sequences = [seq[:SEQ_LENGTH].ljust(SEQ_LENGTH, 'N') for seq in sequences]
    encoded_seqs = [one_hot_encode(seq) for seq in padded_sequences]
    seq_tensor = torch.tensor(np.array(encoded_seqs), dtype=torch.float32).to(device)
    with torch.no_grad():
        raw_embeddings = feature_extractor(seq_tensor).cpu().numpy()
    n_samples = raw_embeddings.shape[0]
    embeddings = raw_embeddings.reshape(n_samples, -1)
    dbscan = DBSCAN(eps=3.5, min_samples=2)
    cluster_labels = dbscan.fit_predict(embeddings)
    results_df = pd.DataFrame({'sequence': sequences, 'cluster_label': cluster_labels})
    return results_df, embeddings

def safe_umap_transform(reducer, emb_sanitized):
        if emb_sanitized.shape[0] == 1:
            # Single sample: compute pairwise distances manually to avoid Numba crash
            distances = pairwise_distances(emb_sanitized, reducer._raw_data, metric=reducer.metric)
            new_coords = reducer._initial_transform(emb_sanitized, distances)
            return new_coords
        else:
            return reducer.transform(emb_sanitized)

def create_umap(
    new_embeddings: np.ndarray,
    cluster_labels: List[int],
    predictions: List[str],
    umap_reducer,
    ref_df,
    distance_threshold
):
    """
    Generates a single UMAP plot colored by model prediction with outliers highlighted,
    and returns it as a Base64 encoded string.
    """
    cluster_centroids = ref_df.groupby('label')[['x', 'y']].mean().to_dict('index')
    # Convert to a format suitable for distance calculation
    centroid_labels = list(cluster_centroids.keys())
    centroid_coords = [list(c.values()) for c in cluster_centroids.values()]

    new_coords = umap_reducer.transform(new_embeddings)

    # --- Step 3: Classify each new point based on distance to centroids ---
    new_labels = []
    for point in new_coords:
        # Calculate distance from the new point to all cluster centroids
        dists = distance.cdist([point], centroid_coords)[0]
        min_dist_idx = np.argmin(dists)
        min_dist = dists[min_dist_idx]
        closest_cluster = centroid_labels[min_dist_idx]

        # Apply the threshold to decide if it's known or novel
        if min_dist <= distance_threshold:
            new_labels.append(f"New_{closest_cluster}")
        else:
            new_labels.append("Potentially Novel")

    # --- Step 4: Prepare DataFrames for plotting ---
    new_df = pd.DataFrame(new_coords, columns=['x', 'y'])
    new_df['label'] = new_labels

    known_new_df = new_df[new_df['label'] != 'Potentially Novel']
    novel_df = new_df[new_df['label'] == 'Potentially Novel']

    # --- Step 5: Plotting ---
    plt.figure(figsize=(14, 12))

    # Plot the original reference map
    ax = sns.scatterplot(
        data=ref_df,
        x='x', y='y',
        hue='label',
        palette=sns.color_palette("hls", len(ref_df['label'].unique())),
        s=20,
        alpha=0.4
    )

    # Plot the newly identified 'known' sequences
    if not known_new_df.empty:
      sns.scatterplot(
          data=known_new_df,
          x='x', y='y',
          hue='label',
          palette='deep', # Use a different palette to make them pop
          s=20, # Large X marker
          ax=ax,
          edgecolor='black',
          linewidth=1
      )

    # Plot the 'novel' sequences as stars
    if not novel_df.empty:
      sns.scatterplot(
          data=novel_df,
          x='x', y='y',
          color='red',
          marker='*',
          s=100, # Very large star marker
          ax=ax,
          label='Potentially Novel',
          edgecolor='black',
          linewidth=1
      )

    plt.title('UMAP with Classified eDNA Samples')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # --- Convert to Base64 ---
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    base64_img = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{base64_img}"

