
# Code Description

![rugby](https://stanfordclubsports.com/images/2020/10/6/largeMR.jpg)

The Classic Post-Rugby Match Social: Half the team is covered in bruises, bandages and black eyes, but that still doesn't mean we can't have fun! Ususally games are played between two teams, including customary games of touch. For these games, we want to ensure that each team has good team chemistry, as well as an equal number of forward players and back players, as to make it a fair and even match up. For that, we ask players to fill out a brief survey before each social event, including name, position and then a brief description of their interests.
The Program will then cluster two teams by interests and position, returning a visualised dataset of who the best teams will be. 

## Sentence Clustering and Top Matches



## How the Code Works

1. **Data Loading**: The code reads the data from the survey, in this case the CSV file named `test.csv`, which contains players' names, their corresponding paragraphs (responses), and their positions (forward or back).

```python
import umap as umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import csv
from scipy import spatial
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from collections import defaultdict

def load_players_map(file_path):
    players_map = {}
    with open(file_path, 'r') as csvfile:
        players = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(players)
        for row in players:
            name, paragraph, position = row
            players_map[paragraph] = name + " " + position
    return players_map

def load_data(file_path):
    forward_data = []
    back_data = []
    with open(file_path, 'r') as csvfile:
        players = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(players)
        for row in players:
            name, paragraph, position = row
            if position == 'forward':
                forward_data.append([name + " " + position, paragraph])
            elif position == 'back':
                back_data.append([name + " " + position, paragraph])
    return forward_data, back_data
```

2. **Sentence Embeddings**: It then utilizes SentenceTransformer, a pre-trained model, to convert each paragraph into a high-dimensional vector representation (embedding). These embeddings capture the semantic meaning of the sentences.

```python def compute_embeddings(model, paragraphs):
    return model.encode(paragraphs)
```

3. **Clustering**: The code employs UMAP (Uniform Manifold Approximation and Projection) and K-means clustering algorithms to group similar embeddings into clusters. It creates two clusters for forward and back positions separately.
```python
def perform_clustering(embeddings):
    reducer = umap.UMAP()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(embeddings)
    reduced_data = reducer.fit_transform(scaled_data)
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_data)
    return reduced_data, cluster_labels
```

4. **Top Matches**: After clustering, the code calculates the top matches for each player based on the cosine similarity of their embeddings. It finds the player whose response is most similar to each player's response, providing valuable insights into players' similarities and differences in their answers.
```python
def calculate_top_matches(players_map, embeddings):
    player_embeddings = {players_map[paragraph]: embedding for paragraph, embedding in embeddings}
    top_matches = {}
    all_personal_pairs = defaultdict(list)
    for player in players_map.values():
        for player1 in players_map.values():
            all_personal_pairs[player].append([spatial.distance.cosine(player_embeddings[player1], player_embeddings[player]), player1])
    for player in players_map.values():
        top_matches[player] = sorted(all_personal_pairs[player], key=lambda x: x[0])

    return top_matches
```
   

5. **Visualization**: The code visualizes the clustered data in a 2D scatter plot using matplotlib. Each player's name is annotated near their corresponding point, allowing for easy interpretation of the clusters.

```python
def plot_clusters(reduced_data, cluster_labels, names):
    x = [row[0] for row in reduced_data]
    y = [row[1] for row in reduced_data]

    plt.figure()
    plt.scatter(x, y, c=cluster_labels, cmap='rainbow', s=60)

    for i, name in enumerate(names):
        plt.annotate(name, (x[i], y[i]), fontsize="6")

    plt.axis('off')
    plt.savefig('equal_teams.png', dpi=800)
```

## Instructions to Run

1. Ensure that Python and the required libraries (umap-learn, matplotlib, scikit-learn, numpy, scipy, sentence-transformers) are installed.

2. Prepare the data in a CSV file (`test.csv`) with the following format: `Name,Paragraph,Position`, where "Name" represents the player's name, "Paragraph" contains their response, and "Position" indicates whether the player is a "forward" or "back."

3. Execute the code in a Python environment, and it will perform clustering, calculate top matches, and generate a scatter plot saved as `equal_teams.png`.

## Example Use Case

This code is helpful in various scenarios, such as team-building exercises, survey analysis, or player evaluation in sports. It enables you to understand how players' responses align and diverge, potentially revealing common patterns and individual characteristics within the group.

Feel free to modify the input data, adjust clustering parameters, or use a different sentence transformer model to suit your specific needs.

Enjoy exploring and gaining valuable insights from your player data with this code! If you have any questions or feedback, please don't hesitate to reach out.

## Visualization of Clusters

![Equal Teams Visualization](Rugby%20Friends/equal_teams.png)
---

*Note: Replace "players" with the appropriate term for the context in which the code is used, e.g., "attendees," "respondents," etc.*










    




if __name__ == "__main__":
    main()
