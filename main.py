import cv2

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_contours(preprocessed_image, threshold_value):
    _, thresh = cv2.threshold(preprocessed_image, threshold_value, 255, cv2.THRESH_BINARY)
    #show threshold image
    cv2.imshow('Threshold', thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #show contours
    cv2.drawContours(preprocessed_image, contours, -1, (0, 255, 0), 2)
    return contours

def rect_overlaps_or_close(rect1, rect2, max_distance):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    close_in_x = abs((x1 + w1 / 2) - (x2 + w2 / 2)) <= max_distance + (w1 + w2) / 2
    close_in_y = abs((y1 + h1 / 2) - (y2 + h2 / 2)) <= max_distance + (h1 + h2) / 2
    return close_in_x and close_in_y

def merge_two_rects(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    x = min(x1, x2)
    y = min(y1, y2)
    w = max(x1 + w1, x2 + w2) - x
    h = max(y1 + h1, y2 + h2) - y
    return (x, y, w, h)

def merge_contours(contours, max_distance):
    rects = [cv2.boundingRect(c) for c in contours]
    merged_rects = []

    while rects:
        rect = rects.pop(0)
        overlap = False

        for idx, other_rect in enumerate(merged_rects):
            if rect_overlaps_or_close(rect, other_rect, max_distance):
                merged_rects[idx] = merge_two_rects(rect, other_rect)
                overlap = True
                break

        if not overlap:
            merged_rects.append(rect)

    return merged_rects

# Load images
image_no_darts = cv2.imread('input/withoutDarts.jpeg')
image_with_darts = cv2.imread('input/withDarts.jpeg')
#resize images
image_no_darts = cv2.resize(image_no_darts, (0,0), fx=0.375, fy=0.375)
image_with_darts = cv2.resize(image_with_darts, (0,0), fx=0.375, fy=0.375)

# Preprocess images
preprocessed_no_darts = preprocess_image(image_no_darts)
preprocessed_with_darts = preprocess_image(image_with_darts)

# Detect contours
contours = detect_contours(cv2.absdiff(preprocessed_no_darts, preprocessed_with_darts), threshold_value=160)

# Merge nearby contours
merged_rects = merge_contours(contours, max_distance=125)

# Draw merged rectangles on the original image
for rect in merged_rects:
    x, y, w, h = rect
    cv2.rectangle(image_with_darts, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the result
cv2.imshow('Darts Detected', image_with_darts)
cv2.waitKey(0)
cv2.destroyAllWindows()
