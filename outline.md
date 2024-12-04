# Quick outline for the paper

1. Abstract
2. Introduction
   1. Background
   2. Motivation
3. Approach
   1. Quick overview of each algorithm
      1. RNN/LSTM
      2. Transformer
      3. Liquid Time-Constant Network
          LTCN is a take on the Recurrant Neural Networks that is based on and inspired by biological neurons, specifically C. Elegens. It uses a process to solve a series of Ordinary Differential Equations (ODEs)
          to solve problems in a way that allows the neural network to act in a spiking fashion, as well as being
          transparent in how it is performing its actions. LTCNs tend to require fewer resources than their deeper
          cousins, and work well with time series data, though they suffer from some of the same issues other RNNs
          suffer from, specifically vanishing gradients, especially with longer sequence data. Some solutions to these
          issues have been made, specifically Closed-form Continuous-time Neural nets, which seeks to reduce the number of Differential Equations down to 1 in an effort to better learn spatiotemporal tasks, and to better learn from sequential data, we can always work with Liquid Structural State-Space Models (Liquid S4).
4. Experiments and Results
5. Conclusion