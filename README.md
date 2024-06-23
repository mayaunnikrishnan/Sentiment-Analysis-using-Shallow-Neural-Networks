# Sentiment-Analysis-using-Shallow-Neural-Networks

## Word2Vec Model Explained
**Overview**
Word2Vec is a popular model used to learn the vector representation of words. Developed by researchers at Google, this model transforms words into numerical vectors such that words with similar meanings are positioned closely together in a vector space. Word2Vec helps capture the contextual meaning of words, which is crucial for tasks in natural language processing (NLP) such as machine translation, sentiment analysis, and more.

**How Word2Vec Works**
Word2Vec uses a neural network to learn word embeddings in one of two ways:

- Continuous Bag of Words (CBOW)
- Skip-Gram
  
1. **Continuous Bag of Words (CBOW)**
CBOW predicts the target word (center word) based on the context of surrounding words. For example, in the sentence "The cat sat on the mat," CBOW tries to predict the word "sat" given the context words "The," "cat," "on," "the," and "mat."

2. **Skip-Gram**
Skip-Gram, on the other hand, does the reverse. It predicts the surrounding context words given the target word. Using the same example, Skip-Gram predicts the words "The," "cat," "on," "the," and "mat" given the word "sat."

**Steps Involved**
- Text Preprocessing
- Tokenize the text into words.
- Remove punctuation and convert text to lowercase.
- Create a vocabulary of unique words.
- Building the Model

**Initialize the neural network with input and output layers.**

- For CBOW, input is the context words, and output is the target word.
- For Skip-Gram, input is the target word, and output is the context words.
  
**Training the Model**

- Use a large corpus of text data.
- Optimize the neural network using stochastic gradient descent or other optimization algorithms.
- Update the weights to minimize the prediction error.

**Generating Word Vectors**

- After training, extract the weights from the hidden layer as the word vectors.
- These vectors capture the semantic relationships between words.
  
**Key Concepts**

- Word Embeddings: Numerical representations of words in a continuous vector space.
- Context Window: The size of the window around the target word to consider context words.
- Negative Sampling: A technique to optimize the training process by simplifying the softmax function.
- Vector Similarity: Words with similar meanings have vectors that are close to each other in the vector space, often measured using cosine similarity.
  
**Applications**

- Semantic Similarity: Finding words with similar meanings.
- Machine Translation: Translating words from one language to another.
- Sentiment Analysis: Understanding the sentiment behind text.
- Recommendation Systems: Suggesting items based on text data.


## GloVe Model Explained

**Overview**
GloVe (Global Vectors for Word Representation) is another popular model used for learning word embeddings. Developed by researchers at Stanford, GloVe aims to capture the global statistical information of words in a corpus. Unlike Word2Vec, which focuses on local context windows, GloVe leverages word co-occurrence statistics from the entire text corpus to learn embeddings.

**How GloVe Works**
GloVe is based on the idea that the meaning of a word can be derived from its co-occurrence with other words. It constructs a co-occurrence matrix where each entry counts how often a pair of words appears together within a certain context window. The objective is to factorize this matrix to obtain word vectors.

**Steps Involved**

**Text Preprocessing**

- Tokenize the text into words.
- Remove punctuation and convert text to lowercase.
- Create a vocabulary of unique words.

**Building the Co-occurrence Matrix**

- Define a context window size (e.g., 5 words to the left and right).
- For each word, count how often it appears with other words within the context window.
- Create a symmetric matrix where each entry (i, j) represents the co-occurrence count of word i and word j.

**Defining the Objective Function**

- The objective is to find word vectors such that their dot product equals the logarithm of the words' co-occurrence probabilities.
- The objective function is designed to minimize the difference between the dot product of word vectors and the log of their co-occurrence counts.

**Training the Model**

- Use gradient descent or other optimization techniques to minimize the objective function.
- Update the word vectors to best approximate the co-occurrence probabilities.

**Key Concepts**

- Co-occurrence Matrix: A matrix where each entry represents how frequently pairs of words appear together.
- Weighting Function: A function used to control the influence of frequent and infrequent word pairs in the learning process.
- Vector Similarity: Words with similar meanings have vectors that are close to each other in the vector space, often measured using cosine similarity.

**Applications**

- Semantic Similarity: Finding words with similar meanings.
- Machine Translation: Translating words from one language to another.
- Sentiment Analysis: Understanding the sentiment behind text.
- Recommendation Systems: Suggesting items based on text data.
