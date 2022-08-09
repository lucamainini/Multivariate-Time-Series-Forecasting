# Multivariate-Time-Series-Forecasting
> using Artificial Neural Networks

## 1. Introduction
The problem we had to face is time series forecasting for multinomial data. To address this task, we used deep learning models with different structures based on LSTM and GRU, and Transformers. 

Our dataset is composed of 68.528 samples of 7 different time series and the aim of the study is to forecast the next 864 points for each feature. 
From a preliminary analysis of the signals [\texttt{Notebook 1}], it came out that they have an almost constant mean and show a periodicity of 96 for all the 7 features. The 7 components have values on very different scales with respect to each other (e.g. Meme creativity $\in [-1.28, 6.06]$ with mean 2.41 and Soap slipperiness $\in [-6.00, 77.37]$ with mean 23.24), hence we needed to apply a normalization (or standardization) before starting to work with the analyses. 
We can detect a certain correlation between the variables \textit{Crunchiness} and \textit{Hype root}, as well as between \textit{Loudness on Impact} and \textit{Wonder Level}. Sometimes, we also have unusual behaviour: most of the values of Crunchiness and Hype Root are positive, but have some negative peaks.

Finally, we have noticed that the data is partially corrupted: there are in fact relatively long sequences of constant data, as shown in Figure 1. This will be an issue that we should address.
<p align="center">
    <img src="./media/constant.png" height="350" alt="constant intervals"/>
    <p align="center">
    Figure 1: An example of constant intervals in the range $[15000, 25000]$.
    
   
