import cv2

# Load the images
image_no_darts = cv2.imread('input/withoutDarts.jpeg')
image_with_darts = cv2.imread('input/withDarts.jpeg')
#resize the images
image_no_darts = cv2.resize(image_no_darts, (500, 500))
image_with_darts = cv2.resize(image_with_darts, (500, 500))
#show the images
cv2.imshow('No Darts', image_no_darts)
cv2.imshow('With Darts', image_with_darts)

# Ensure both images are the same size
image_no_darts = cv2.resize(image_no_darts, (image_with_darts.shape[1], image_with_darts.shape[0]))

gray_no_darts = cv2.cvtColor(image_no_darts, cv2.COLOR_BGR2GRAY)
gray_with_darts = cv2.cvtColor(image_with_darts, cv2.COLOR_BGR2GRAY)

blurred_no_darts = cv2.GaussianBlur(gray_no_darts, (5, 5), 0)
blurred_with_darts = cv2.GaussianBlur(gray_with_darts, (5, 5), 0)

difference = cv2.absdiff(blurred_no_darts, blurred_with_darts)
cv2.imshow('Difference', difference)

_, thresh = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
cv2.imshow('Thresh', thresh)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for c in contours:
    # You can add more conditions to filter out noise
    if cv2.contourArea(c) > 150:  # min_area should be set based on your observations
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(image_with_darts, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('Darts Detected', image_with_darts)
cv2.waitKey(0)
cv2.destroyAllWindows()