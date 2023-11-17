import cv2

# Load an image
image_path = 'input\DartVonSchrank.jpeg'
image = cv2.imread(image_path)
image = cv2.resize(image, (0, 0), fx=0.375, fy=0.375)

# Display the image
cv2.imshow('Dartboard', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def detect_darts(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #show the gray image to debug
    cv2.imshow('gray', gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #show the blurred image to debug
    cv2.imshow('blurred', blurred)
    edged = cv2.Canny(blurred, 50, 200)
    #show the edged image to debug
    cv2.imshow('edged', edged)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours
    for c in contours:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # If the shape is elongated, it could be a dart
        if len(approx) > 3 and len(approx) < 5:  # Adjust based on your observations
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if aspect_ratio > 1.2:  # Adjust this value based on the orientation and shape of the darts
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

    return image

# Process the image and show the result
processed_image = detect_darts(image)
cv2.imshow('Dart Detection', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()