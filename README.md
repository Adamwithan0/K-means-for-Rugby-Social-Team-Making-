## Visualization of Clusters

![Equal Teams Visualization](equal_teams.png)

# Code Description

## Sentence Clustering and Top Matches

This code performs sentence clustering and identifies top matches among a group of players based on their responses in a given test or survey. The code uses sentence embeddings and dimensionality reduction techniques to group similar responses together, allowing for insightful analysis and comparison of players' answers.

## How the Code Works

1. **Data Loading**: The code reads data from a CSV file named `test.csv`, which contains players' names, their corresponding paragraphs (responses), and their positions (forward or back).

2. **Sentence Embeddings**: It utilizes SentenceTransformer, a pre-trained model, to convert each paragraph into a high-dimensional vector representation (embedding). These embeddings capture the semantic meaning of the sentences.

3. **Clustering**: The code employs UMAP (Uniform Manifold Approximation and Projection) and K-means clustering algorithms to group similar embeddings into clusters. It creates two clusters for forward and back positions separately.

4. **Top Matches**: After clustering, the code calculates the top matches for each player based on the cosine similarity of their embeddings. It finds the player whose response is most similar to each player's response, providing valuable insights into players' similarities and differences in their answers.

5. **Visualization**: The code visualizes the clustered data in a 2D scatter plot using matplotlib. Each player's name is annotated near their corresponding point, allowing for easy interpretation of the clusters.

## Instructions to Run

1. Ensure that Python and the required libraries (umap-learn, matplotlib, scikit-learn, numpy, scipy, sentence-transformers) are installed.

2. Prepare the data in a CSV file (`test.csv`) with the following format: `Name,Paragraph,Position`, where "Name" represents the player's name, "Paragraph" contains their response, and "Position" indicates whether the player is a "forward" or "back."

3. Execute the code in a Python environment, and it will perform clustering, calculate top matches, and generate a scatter plot saved as `equal_teams.png`.

## Example Use Case

This code is helpful in various scenarios, such as team-building exercises, survey analysis, or player evaluation in sports. It enables you to understand how players' responses align and diverge, potentially revealing common patterns and individual characteristics within the group.

Feel free to modify the input data, adjust clustering parameters, or use a different sentence transformer model to suit your specific needs.

Enjoy exploring and gaining valuable insights from your player data with this code! If you have any questions or feedback, please don't hesitate to reach out.

---

*Note: Replace "players" with the appropriate term for the context in which the code is used, e.g., "attendees," "respondents," etc.*
