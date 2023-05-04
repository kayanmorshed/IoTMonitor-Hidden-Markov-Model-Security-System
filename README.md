# IoTMonitor: A Hidden Markov Model-based Security System for IoT Network

This repository contains the Python implementation of a **Hidden Markov Model (HMM)** based security solution to determine underlying sequential patterns in a large dataset of sensory data collected from diverse IoT devices. The goal of this project is to discover probabilistic relations among smart devices in an IoT network, determine network dynamics, and extract optimal hidden sequence an agent may follow to reach a target node in the network. It is assumed that an attacker actively tries to compromise an IoT network, and the implemented solution exploits attacker's behavior to defend against the attack.  

In this project, two prominent algorithms are implemented from scractch using Python:  
1. [Baum Welch Algorithm](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm): to estimate _probability state distribution_, _state transition probabilities_, and _emission probabilities_. 

2. [Viterbi Algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm): to extract the optimal hidden sequence. 

## Dataset

**Dataset Name**: Peeves: Physical Event Verification in Smart Homes 

**Citation**: Birnbach, S., & Eberz, S. (2019). Peeves: Physical Event Verification in Smart Homes. University of Oxford.

**Link**: https://ora.ox.ac.uk/objects/uuid:75726ff7-fee1-420d-8a17-de9572324c7d 

**About the dataset**: https://ora.ox.ac.uk/objects/uuid:75726ff7-fee1-420d-8a17-de9572324c7d/download_file?file_format=pdf&safe_filename=readme.pdf&type_of_work=Dataset).


## Codebase Information

**Language used**: Python

**Libraries used for data analytics**: [NumPy](https://numpy.org/doc/stable/index.html), [pandas](https://pandas.pydata.org), [scikit-learn](https://scikit-learn.org/stable)

**Libraries used for visualization**: [Matplotlib](https://matplotlib.org), [Seaborn](https://seaborn.pydata.org)


## Usage

1. **Install dependencies**: Use the following commands to install required dependencies:

```
python -m pip install -U pip # to install pip
pip install numpy
pip install pandas
python -m pip install -U matplotlib
pip install -U scikit-learn
pip install seaborn
```

2. **Run project**: Once all the necessary dependencies are installed on your machine, simply execute the `hmm-main.py` file. It will import all necessary modules to successfully run the application. 


## Published Research Paper: 

1. **Conference Version**: https://ieeexplore.ieee.org/abstract/document/9771878 

2. **Pre-print Version**: https://github.com/kayanmorshed/IoTMonitor-Hidden-Markov-Model-Security-System/blob/main/IoTMonitor_Alam_2022.pdf

