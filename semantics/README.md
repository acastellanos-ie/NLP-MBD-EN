# Semantics

## Overview
We include different notebooks to provide the student with practical means to explore some of the modern methods used to represent meaning in NLP and NLU. We follow an incremental approach in accordance with the contents of session 5 of the course, focused on semantics.

## Notebooks

### Word embeddings
Our [first notebook](https://github.com/hybridnlp/tutorial/blob/master/01_capturing_word_embeddings.ipynb) is focused on learning word embeddings from a training corpus, assigning a vector to each word, following the distributional representational theory. We will see how to use one of the many algorithms available nowadays to this purpose and illustrate some of the properties that such word vector representations have, allowing for basic semantic operations, like checking word similarity, relatedness and analogy based on the cosine distance between the vectors that represent such words in the embeddings space.

### Knowledge Graph Embeddings
Word embeddings aim at capturing the meaning of words based on large corpora; however, there are decades of experience and approaches that have tried to capture meaning by structuring knowledge into semantic networks, ontologies and graphs. Examples of such resources include lexical knowledge graphs like WordNet. The [second notebook](https://github.com/hybridnlp/tutorial/blob/master/02_knowledge_graph_embeddings.ipynb) shows how to leverage such resources and calculate representations of their content, like lemmas, senses, and the relations between them, in a common vector space.

### Hybrid representations from text corpora and knowledge graphs
Knowledge graph embedding algorithms can capture structured knowledge about concepts and relations in a graph as embeddings in a vector space, which then can be used in downstream tasks. However, this type of approaches can only capture the knowledge that is explicitly represented in the graph, hence lacking in recall and domain coverage. Here we include toy examples of algorithms that address this limitation through the combination of information from both unstructured text corpora and structured knowledge graphs. The [third notebook](https://github.com/hybridnlp/tutorial/blob/master/03_vecsigrafo.ipynb) shows how to combine document corpora and knowledge graphs to learn richer representations that better capture the meaning of words.

## Additional resources and bilbiography
This notebooks are part of the tutorial [Hybrid Techniques for Hybrid Natural Language Processing](http://hybridnlp.expertsystemlab.com/tutorial/). The complete lessons, materials and resources of the tutorial can be found [here](http://hybridnlp.expertsystemlab.com/tutorial/index.php/sample-page/). You can find all the notebooks of the tutorial in its [repo](https://github.com/hybridnlp/tutorial). The following book is focused on the concepts illustrated here: _Jose Manuel Gomez-Perez, Ronald Denaux and Andres Garcia-silva. 2020. A Practical Guide to Hybrid Natural Language Processing - Combining Neural Models and Knowledge Graphs for NLP. https://doi.org/10.1007/978-3-030-44830-1_

