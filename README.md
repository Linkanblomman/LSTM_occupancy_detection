# PyTorch LSTM occupancy detection model
*Long Short Term Memory* occupancy detection in smart homes using indoor climate data

### Paper
To read about this project: 
[LSTM for occupancy detection in smart homes using indoor climate data](/Images/LSTM_for_occupancy_detection_in_smart_homes_using_indoor_climate_data.pdf)

#### Datasets from the paper
* [Occupancy detection dataset](https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+)
* [Room climate dataset](https://github.com/IoTsec/Room-Climate-Datasets)

---

## Result

LSTM predictions on the *Occupancy detection dataset*

![LSTM predictions](/Images/Rasp_terminal.png)

---

### How to run it

**Run code**

1. Choose a notebook from Github
2. Press the ![Colab button](/Images/colab_button.jpg) button then run it

**Or**

1. Go to [Google Colab](https://colab.research.google.com) and sign in
2. Open "*Open Notebook*" then go to "*GitHub*" tab and then search for "*Linkanblomman*" and choose repository "*Linkanblomman/LSTM_occupancy_detection*"
3. Pick the notebook and run it

**Run on a Raspberry Pi 3**
1. Download the LSTM_Raspberry_Pi directory to your Raspberry Pi 3
2. Follow this guide: [A Step by Step guide to installing PyTorch in Raspberry Pi](https://medium.com/secure-and-private-ai-writing-challenge/a-step-by-step-guide-to-installing-pytorch-in-raspberry-pi-a1491bb80531) 
3. To get necessary files, run [Raspberry_Pi_3_Occupancy_Detection_Dataset_LSTM_model.ipynb](https://github.com/Linkanblomman/LSTM_occupancy_detection/blob/main/Raspberry_Pi_3_Occupancy_Detection_Dataset_LSTM_model.ipynb)
4. Extract the zip file from step 3 into the LSTM_Raspberry_Pi directory from step 1
5. Pip install necessary modules
6. Run code as the image below shows with the command "Python3 lstm__for_raspberrypi.py"

![Rasp terminal](/Images/Rasp_terminal.png)


**NOTE!**

Before running the code in Google Colab.

**GPU usage** (change CPU to GPU): Runtime -> Change runtime type -> Hardware accelerator -> GPU -> Save

If you get errors when running [Raspberry_Pi_3_Occupancy_Detection_Dataset_LSTM_model.ipynb](https://github.com/Linkanblomman/LSTM_occupancy_detection/blob/main/Raspberry_Pi_3_Occupancy_Detection_Dataset_LSTM_model.ipynb):
1. Press the **"RESTART RUNTIME"**
2. Run again: Runtime -> Run all
3. Still getting errors then repeat step 1 and 2 again.
