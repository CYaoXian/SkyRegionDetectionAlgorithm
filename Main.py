# Import libraries
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

# Function for canny edge detection algorithm
def cannySkyRegionDetection(img):
    # Extract blue channel from input imgae
    bluePlane = img[:, :, 0]
    # Apply canny edge detection algorithm on the bluePlane to extract the edges
    cannyEdges = cv2.Canny(bluePlane, 16, 186)
    
    # Create a 9x9 rectangular structring element for morphological operations
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    # Perform morpgological closing
    cannyEdgesClosing = cv2.morphologyEx(cannyEdges, cv2.MORPH_CLOSE, se)

    # Function for floodfill algorithm
    def performFillHole (imgInput): 
        # Create a copy of input image
        floodFillHole = imgInput.copy()
        # Obtain height and width of the image
        h, w = imgInput.shape[:2]
        # Create mask for floodfill
        mask = np.zeros((h + 2, w + 2), np.uint8)
        # Convert datatype to uint8
        floodFillHole = floodFillHole.astype("uint8")
        # Perform floodfill operation from point (50,0)
        cv2.floodFill(floodFillHole, mask, (50, 0), 255)
        # Invert floodfilled image
        imgInvert = cv2.bitwise_not(floodFillHole)
        # Combine the original image with inverted image using bitwise OR operation
        floodFillImgOut = imgInput | imgInvert  
        return floodFillImgOut 
    
    holesFilled = performFillHole(cannyEdgesClosing)
    invertedImg = cv2.bitwise_not(holesFilled)
    
    # Return the inverted image 
    return invertedImg

# Function for sky line detection
def skylineDetection (mask):
    # Apply canny edge detection algorithm on the mask image using 30 low threshold and 200 high threshold
    skylineDec = cv2.Canny(mask, 30, 200)
        
    return skylineDec


# Function for detecting day or night time
def dayOrNightTime(img, threshold=100):
    
    # Convert the image to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculate the mean intensity of the image
    meanIntensity = cv2.mean(imgGray)[0]
    # Compare with the threshold to classify as day or night
    if meanIntensity > threshold:
        return 'Day'
    else:
        return 'Night'
   
# Function for detecting accuracy using Mean absolute error
def meanAbsoluteError(mask, img):
    
    # Ensure both images have the same shape
    assert mask.shape == img.shape, 'Image shapes do not match.'

    # Calculate the absolute difference between the two images
    absoluteDiff = np.abs(mask - img)
    
    # Calculate the mean absolute error
    mae = np.mean(absoluteDiff)
    return mae

# Main function
def main(skyDataSetsNum, fileName):
    # Read the datasets images 
    img = cv2.imread(f'SkyFinderDatasets\{skyDataSetsNum}\{fileName}', 1)
    # Perform Canny edge detection algorithm to obtain the predicted mask of the sky region
    maskResult = cannySkyRegionDetection(img)
    
    # Load the expected mask image for dataset '623', '684', '9730', '10917'
    if skyDataSetsNum == '623': 
        expectedMask = cv2.imread('mask/623.png', 0)

    elif skyDataSetsNum == '684': 
        expectedMask = cv2.imread('mask/684.png', 0)

    elif skyDataSetsNum == '9730': 
        expectedMask = cv2.imread('mask/9730.png', 0)

    elif skyDataSetsNum == '10917': 
        expectedMask = cv2.imread('mask/10917.png', 0)
        
    # Use Canny edge detection algorithm on the predicted mask to detect the skyline
    skylineDec = skylineDetection(maskResult)
    # Determine whether the image is a day or night image based on the mean intensity
    dayNight = dayOrNightTime(img, 108)

    # # Convert bgr to rgb
    rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(rgbImg)

    plt.subplot(1, 3, 2)
    plt.title('Mask')
    plt.imshow(maskResult, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Skyline Image')
    plt.imshow(skylineDec, cmap='gray')

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Get the current working directory
    currentDirectory = os.getcwd()
    # Create a new directory named "Result" within the current directory to store the results
    resultDirectory = os.path.join(currentDirectory, 'test')
    if not os.path.exists(resultDirectory):
        os.makedirs(resultDirectory)
    
    # Create separate subdirectories for day and night images based on dataset number
    if dayNight == 'Day':
        subdir = 'Day'
    elif dayNight == 'Night':
        subdir = 'Night'
        
    datasetDirectory = os.path.join(resultDirectory, f"{skyDataSetsNum}/{subdir}")
    if not os.path.exists(datasetDirectory):
        os.makedirs(datasetDirectory)
    
    # Save the figure in the respective dataset and day/night subdirectory
    plt.savefig(os.path.join(datasetDirectory, f'{fileName}.jpg'))
       
    if dayNight == 'Day':
        # Calculate the accuracy for day images based on whether it's day or night
        dayAccuracy = 1 - meanAbsoluteError(expectedMask, maskResult) / 255  
        # Set nightAccuracy to None to avoid UnboundLocalError
        nightAccuracy = 1  
    elif dayNight == 'Night':
        # Calculate the accuracy for night images based on whether it's day or night
        nightAccuracy = 1 - meanAbsoluteError(expectedMask, maskResult) / 255
        # Set nightAccuracy to None to avoid UnboundLocalError
        dayAccuracy = 1
        
    plt.close()
    return dayAccuracy, nightAccuracy

    
if __name__ == '__main__':
    # Create empty lists to store the individual day and night accuracies for each dataset
    listTotalDayAccuracy = []
    listTotalNightAccuracy = []
    # Get the current working directory
    currentDirectory = os.getcwd()
    # Define a list of dataset numbers to be processed
    skyDataSetsNum = ['623', '684', '9730', '10917']
    
    # Loop through each dataset in the list
    for data in skyDataSetsNum:
        # Create empty lists to store the day and night accuracies for each image in the current dataset
        listDayAvg = []
        listNightAvg = []
        # Form the path to the current dataset directory
        datasetDirectory = os.path.join(currentDirectory, f'SkyFinderDatasets\{data}')
        
        # Loop through each image file in the current dataset directory
        for fileName in os.listdir(datasetDirectory):
            # Call the main function to get the day and night accuracies for the current image
            dayAccuracy, nightAccuracy = main(data, fileName)
            
            # Append the day and night accuracies to their respective lists
            listDayAvg.append(dayAccuracy)
            listNightAvg.append(nightAccuracy)

        # Calculate the mean day and night accuracies for the current dataset
        skyDatasetDayAcc = np.mean(listDayAvg)
        skyDatasetNightAcc = np.mean(listNightAvg)
        
        # Append the mean day and night accuracies to the total accuracy lists
        listTotalDayAccuracy.append(skyDatasetDayAcc)
        listTotalNightAccuracy.append(skyDatasetNightAcc)
        
        # Print the mean day and night accuracies for the current dataset
        print(f'Day Dataset {data}: ', skyDatasetDayAcc)
        print(f'Night Dataset {data}: ', skyDatasetNightAcc)
        
    # Calculate the total mean day and night accuracies across all datasets
    totalDayAccuracy = np.mean(listTotalDayAccuracy)
    totalNightAccuracy = np.mean(listTotalNightAccuracy)
    
    # Print the total mean day and night accuracies
    print('Total Day: ', totalDayAccuracy)
    print('Total Night: ', totalNightAccuracy)
