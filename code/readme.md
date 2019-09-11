# Tensorflow-UEGazeNet
A tensorflow based gaze-tracking convolutional neural network model.

# Files
data_source.py: The data source of UnityEyes.

UT_mutliviews.py: The data source of UT multiviews.

MPIIGaze.py: The data source of MPIIGaze.

make_gestures_dataset.py: Make the gestures dataset, the image size is 30 × 30.

draw_gesture.py: Track gaze and draw the gaze gestrue.

gesture_model.py: Gaze gesture was classified by KNN, RF, SVM and XGBoost respectively.

gestures_cnn.py: Classify the gaze gestures by CNN and ANN.

UE.py: Train the model with UnityEyes.

UT.py: Train the model with UT multiviews.

utils.py: Some functions we used.

# Files directory
```
UEGazeNet
|---imgs_UnityEyes
|   |---0.jpg
|   |---0.json
|   |---1.jpg
|   |---1.json
|   └---...
|---imgs_UT_Multiviews
|   └---UT
|       |---data
|       |   |---s00
|       |   |   |---raw
|       |   |   |---synth
|       |   |   |   |---000_left.csv
|       |   |   |   |---000_left.zip
|       |   |   |   |---000_right.csv
|       |   |   |   |---000_right.zip
|       |   |   |   └---...
|       |   |   └---test
|       |   |---s01
|       |   └---...
|       └---temp
|           |---0.jpg
|           |---0.txt
|           └---...
|---imgs_MPIIGaze
|   └---Data
|       |---temp
|       |   |---0.jpg
|       |   |---0.txt
|       |   └---...
|       └---Normalized
|           |---p00
|           |   |---day01.mat
|           |   └---...
|           └---...
|---gesture_data
|   |---all
|   |   |---0 # the number of each pattern
|   |   |   |---0_L.jpg
|   |   |   |---0_R.jpg
|   |   |   └---...
|   |   └---...
|   |---data.npy # dataset with image size 30 × 30
|   |---0 # the number of each people
|   |   |---first
|   |   |   |---0_L.jpg
|   |   |   |---0_R.jpg
|   |   |   └---...
|   |   └---second
|   └---...
|---test_acc
|   |---acc.jpg
|   |---indoor
|   └---outdoor
|---test # some data for test
|---params # Saved model parameters
|---data_source.py
|---UT_mutliviews.py
|---MPIIGaze.py
|---make_gestures_dataset.py
|---draw_gesture.py
|---gesture_model.py
|---gestures_cnn.py
|---UE.py
|---UT.py
|---utils.py
```







# Details for draw gesture

**Directory where the files saved'./gesture_data/'
**The model'./params/'
**Naming format  0_L（R）.jpg
**The number of each run starts from 0, so the data should be moved to the corresponding location to save at the end of each run
------

#### 1.Adjust the position so that it can get a full view of both eyes 

#### 2.Record the gaze vectors of the four top corners of the screen in turn
* Top left corner of record -----------'a'
* record upper right hand corner ------'s'
* record lower left corner ------------'z'
* record the lower right corner -------'x'
    
    
#### 3. Choose to start tracking or stop
* start tracking ------'c'
* stop tracking -------'v'
##### After stops tracking, it is best to record four top angles again before the next tracking 
        
#### 4. Select drawing mode
* enter drawing mode ------'q'
##### keeps tracking of gaze direction after entering this mode, which is equivalent to drawing lines all the time, instead of showing only 10 frames of gaze direction as in normal mode
        
#### 5. Select save drawing or redraw drawing
* save the drawing ------'w'
##### It will exit the drawing mode after each save, in order to adjust the gaze direction for the next drawing
* cancel drawing ------'e'
##### when the drawing is not satisfactory, you can exit the drawing mode, in order to adjust the gaze direct for the next drawing
        
#### 6. End the tracking
* stop tracking ------'v'
* end the program ------'p'

## environment
**python3.6     tensorflow1.8.0**  
**numpy      pykeyboard      pymouse    opencv**    **pyautogui**
