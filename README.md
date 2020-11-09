# extreme-learning-machines

## Aim
Develop a neural decoder based on Extreme Learning Machines (ELM) for the P300 paradigm for single trial prediction at the population level

## Motivation
Brain computer interfaces (BCIs) can provide a direct communcation channel between the human brain and external devices. Patients with inhibited motor control over their muscles, such as locked-in patients, can utilize BCIs to communicate through their thoughts. 

The P300 paradigm occurs when two events are experience by sound or sight and one is more rare than the other. A positive component is generated in the neural signals (EEG traces) about 300 milliseconds after experiencing the rare event. P300 BCI systems take advantage of this phenomena for spelling, cursor and wheelchair movements.

Currently machine learning algorithms for BCI systems are still:
- computationally time consuming: training tends to depend on iterative learning processes
- require prior knowledge: the P300 component and neural signals in general are highly variable
- are to be taxing to the user during the learning process: multiple trials are needed for training in order to obtain good accuracies

Extreme learning machines (ELMs) were explored for neural decoding as it has been shown to have fast training (no iterative learning) and good generalization. Taking advantage of these properties, ELM networks were trained using population features and predictions were made on single trials.

## Strategy
1. Use population features to built ELM networks for detection of the P300 component from EEG data
2. Assess performance in regards to time, accuracy, recall and precision
3. Built ensemble ELM classifiers: simple majority voting, bagging and boosting
4. Compare performance of single ELM networks vs ensemble ELM classifiers
