The goal of the code in this folder is the following: After we realized that there is a strong correlation between radiation strength (coming from the sun) and the temperature sensors located on the south of the Matterhorn (even for temperature measures below the rock surface), we hypothesized that the temperature change below surface can be predicted from knowing the temperature above the surface and the weather conditions. 

As weather conditions, we would consider the radiation strength as well as images (which include the sensor location in their field of view and thus can be used to detect occlusions of radiation onto the sensor spot).

The ultimate goal would have been, to train a Bayesian neural network on this data and to investigate the images that lead (together with the other information) to uncertain predictions. I.e., under what weather conditions does the strong correlation between radiation and below surface temperature not hold.

Future steps could be to investigate the influence of humidity data and or wind data on the predictive model.

### Implementation

We implemented a PyTorch dataset that can represent the data as we need it for the above task. Additionally, we use a ResNet-32 to process the images. To condition the resnet on the timeseries data (e.g., radiation, above-surface temperature, ...), we use a simple fully-connected network that generates the batch normalization weights of the resnet.

Since temperature measurements don't always have a corresponding image, we only took those measurements that have an image within a 20minute window around the measurement.
