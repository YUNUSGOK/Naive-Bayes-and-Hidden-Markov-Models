
# Homework 4 - Naive Bayes, Hidden Markov Models

This project contains naive bayes and hidden markov model implementation in python3.

## Naive Bayes

The code reads data from hw4_data and trains the model with train data. After training it calculates using test_data and labels. 
### To run


```bash
python3 nb.py
```
* Output
```bash
Accuracy: 0.7609289617486339
```

## Hidden Markov Models
* Evaluation

[Forward algorithm](https://en.wikipedia.org/wiki/Forward_algorithm) is implemented to "forward" function in hmm.py 
* Decoding 

[Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm) is implemented to "viterbi" function in hmm.py 

```bash
python3 hmm_mini_test.py
```
