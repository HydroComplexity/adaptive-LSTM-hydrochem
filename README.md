https://doi.org/10.5281/zenodo.16755028

Summary: Stream water carries dissolved solutes, influenced by natural events like rain, snow-melt, and drought as well as human activities. Understanding how solute concentrations change over time is crucial for managing water quality. Rapid changes in solute concentrations are often caused by rain or snow-melt carrying solutes from the land into rivers through different pathways. While substantial progress has been made in understanding both long-term trends and rainfall event-scale solute dynamics, predictive models still face challenges in accurately capturing rapid, short-term fluctuations in water chemistry at high temporal resolutions. This study uses high-frequency solute concentration data from two watersheds in the U.S. and France to explore the predictability of solute concentrations under all conditions including a prolonged drought in central Illinois, and at event timescales. We enhance a machine learning model, known as Long Short-Term Memory (LSTM), by incorporating flow and flux gradients within the model architecture. These enhancements help the model better account for rapid changes in solute concentrations and hysteresis behaviors. Further, we develop an adaptive version of the model that applies a given modification based on flow and flux gradients relative to thresholds, enabling it to capture solute variability under different conditions. These findings help us better utilize the distinct physical processes that influence water chemistry for prediction, offering valuable tools for improving water quality and conservation efforts. By combining high-resolution data with advanced machine learning techniques, this work highlights new ways to address the challenge of water quality predictions.

File Overview: 
1. lstm_classes.py: Contains all the classes, functions for LSTM model. 
2. Org_adptLSTM.py: To run all the individual augmentation in the LSTM as well as adaptive LSTM model for Orgeval, France data. There are index which can be selected to turn on and off individual gate modulations in LSTM.  
3. USRB_adptLSTM.py: To run all the individual augmentation in the LSTM as well as adaptive LSTM model for USRB, Monticello, Illinois, USA data. There are index which can be selected to turn on and off individual gate modulations in LSTM.  
4. plot_org.py: Plots all the results from Orgeval, France data.
5. plot_USRB.py: Plots all the results from USRB, Monticello, Illinois, USA data.

Python versions used:

1. python: 3.10.13
2. cuda: 12.4
3. pytorch: 1.13.0
4. numpy: 1.24.4
5. matplotlib: 3.7.2
6. pandas: 2.1.4
7. scipy: 1.8.0
8. scikit_learn: 1.3.0
9. seaborn: 0.13.2
10. tqdm: 4.65.0
