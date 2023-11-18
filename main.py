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

min_area = 0  # Adjust as needed

# Ensure both images are the same size
image_no_darts = cv2.resize(image_no_darts, (image_with_darts.shape[1], image_with_darts.shape[0]))

gray_no_darts = cv2.cvtColor(image_no_darts, cv2.COLOR_BGR2GRAY)
gray_with_darts = cv2.cvtColor(image_with_darts, cv2.COLOR_BGR2GRAY)
difference = cv2.absdiff(gray_no_darts, gray_with_darts)

_, thresh = cv2.threshold(difference, 200, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    # You might still need to filter based on size or shape
    if cv2.contourArea(c) > min_area:  # min_area as determined earlier
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(image_with_darts, (x, y), (x + w, y + h), (0, 255, 0), 2)

#combine bigger clusters of contours into one

cv2.imshow('Darts Detected', image_with_darts)
cv2.waitKey(0)
cv2.destroyAllWindows()

#print cords of darts
for c in contours:
    # You might still need to filter based on size or shape
    if cv2.contourArea(c) > min_area:  # min_area as determined earlier
        (x, y, w, h) = cv2.boundingRect(c)
        print("Dart at: " + str(x) + ", " + str(y))