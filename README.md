# Homophily Bias in Graph Neural Networks: Limitations of Graph Convolution for Node Classification

This is the repository for CS514 project "Homophily Bias in Graph Neural Networks: Limitations of Graph Convolution for Node Classification"

### Datasets

We conducted our experiments on commonly used heterophilous datasets, which include squirrel, chameleon, texas, cornell, and wisconsin. These datasets are frequently referenced in heterophily research as they exhibit varying degrees of heterophilous characteristics and structural properties that challenge conventional graph neural network (GNN) models. Specifically:

#### Roman-empire
This dataset, derived from the Roman Empire article on English Wikipedia (March 1, 2022 dump), represents each (non-unique) word as a node. Edges connect sequential words or those linked via the sentence dependency tree, forming a chain-like structure with shortcut edges for syntactic dependencies. Node classes correspond to syntactic roles, with the 17 most frequent roles treated as distinct classes and others grouped into an 18th class, identified using spaCy. Node features utilize fastText embeddings. This dataset evaluates GNN performance in low homophily and sparse connectivity scenarios. The graph contains 22.7K nodes, 32.9K edges, an average degree of 2.9, and a diameter of 6824. It exhibits heterophily ($h_{adj} = -0.05$) and high label informativeness compared to other datasets, reflecting complex label connectivity patterns.

#### Amazon-ratings
Based on the Amazon co-purchasing network , this dataset models products (books, music CDs, DVDs, VHS tapes) as nodes, with edges linking products frequently purchased together. The task involves predicting a product’s average rating (grouped into five classes). Node features are derived from the mean of fastText embeddings for product descriptions. To reduce complexity, only the largest connected component of the graph’s 5-core is used.

#### Minesweeper
This synthetic dataset mimics a Minesweeper game with a 100x100 grid graph, where nodes (cells) connect to up to eight neighbors, except at the edges. Approximately 20% of nodes are designated as mines, and the task is to identify these. Node features include one-hot-encoded counts of neighboring mines, with binary flags indicating unknown features for 50% of nodes. The graph’s regular structure results in an average degree of 7.88 and near-zero adjusted homophily and label informativeness due to the random placement of mines.

#### Tolokers
Constructed from the Toloka crowdsourcing platform, this dataset represents workers as nodes and links those who collaborated on tasks. The task predicts which workers were banned from projects. Node features include worker profile and performance statistics. The graph contains 11.8K nodes with an average degree of 88.28, making it the densest in the benchmark, with 22% of nodes corresponding to banned workers.

#### Questions
Derived from Yandex Q, this dataset represents users as nodes, with edges linking users who answered each other’s questions within a year (September 2021–August 2022). Focused on the "medicine" topic, the task predicts whether users remained active (not deleted or blocked). Node features are derived from the mean of fastText embeddings of user descriptions, with binary flags for missing descriptions (15\%). The graph has 48.9K nodes, an average degree of 6.28, and exhibits high edge homophily but heterophily overall ($h_{adj} = 0.02$). It also has the lowest clustering coefficients, indicating minimal closed node triplets.

#### Squirrel and Chameleon
Nodes in these datasets represent Wikipedia articles, with edges based on mutual links. Classes are defined based on traffic levels, and node features encode the presence of particular terms. These datasets contain duplicate nodes, which we retained in line with standard practices to allow for direct comparison with previous work.

#### Texas, Cornell, and Wisconsin
These datasets represent web pages from the WebKB network, where nodes correspond to pages and edges to hyperlinks. Node features are derived from a bag-of-words representation, with classification targets corresponding to categories such as "student," "course," or "faculty." These datasets have fewer nodes and edges than Wikipedia networks, offering a contrasting perspective due to their smaller and more imbalanced class distributions.
In our experiments, we use standard train/validation/test splits, ensuring consistency with previous studies to enable fair comparison of model performance. 

### Running experiments

In our report, we introduce a set of novel homophily measures designed to address the limitations of existing metrics and provide a more comprehensive understanding of node and edge relationships in heterophilous graph structures. While traditional measures such as edge homophily, node homophily, and class homophily provide valuable insights, they often lack the flexibility to capture nuanced relationships between similarity measures. Our proposed metrics aim to bridge this gap by extending and generalizing these concepts..

To reproduce results of our baseline models (ResNet and standard GNNs), you need to install [PyTorch](https://pytorch.org/) and [DGL](https://www.dgl.ai/pages/start.html) (see the full list of requirements in `enivronment.yml`). Then you can run `scripts/run_1/2/3/4/5.sh` (each script contains different datasets). After that you can view a table with results in `notebooks/results.ipynb`.
