import cv2
import numpy as np
import pyttsx3

# Initialize pyttsx3 engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speech rate if needed

# Load reference images
reference_50 = cv2.imread('fifty_peso.jpg', 0)
reference_100 = cv2.imread('hundred_peso.jpg', 0)

# Resize reference images
reference_50_resized = cv2.resize(reference_50, (0, 0), fx=0.5, fy=0.5)
reference_100_resized = cv2.resize(reference_100, (0, 0), fx=0.5, fy=0.5)

# ORB feature detector
orb = cv2.ORB_create(nfeatures=5000)

kp_50, des_50 = orb.detectAndCompute(reference_50_resized, None)
kp_100, des_100 = orb.detectAndCompute(reference_100_resized, None)

# FLANN matcher
index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

cap = cv2.VideoCapture(0)

# Flags to avoid repeated speech announcements
announced_50 = False
announced_100 = False

while True:
    ret, scene_image = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray_scene = cv2.cvtColor(scene_image, cv2.COLOR_BGR2GRAY)
    gray_scene = cv2.equalizeHist(gray_scene)

    kp_scene, des_scene = orb.detectAndCompute(gray_scene, None)

    # Match features with FLANN
    matches_50 = flann.knnMatch(des_50, des_scene, k=2)
    matches_100 = flann.knnMatch(des_100, des_scene, k=2)

    # Apply Lowe's ratio test for good matches
    good_matches_50 = []
    good_matches_100 = []

    for match in matches_50:
        if len(match) == 2:
            m, n = match
            if m.distance < 0.6 * n.distance:
                good_matches_50.append(m)

    for match in matches_100:
        if len(match) == 2:
            m, n = match
            if m.distance < 0.6 * n.distance:
                good_matches_100.append(m)

    # Detect and draw 50 Peso Bill
    if len(good_matches_50) > 10:
        src_pts_50 = np.float32([kp_50[m.queryIdx].pt for m in good_matches_50]).reshape(-1, 1, 2)
        dst_pts_50 = np.float32([kp_scene[m.trainIdx].pt for m in good_matches_50]).reshape(-1, 1, 2)

        M_50, mask_50 = cv2.findHomography(src_pts_50, dst_pts_50, cv2.RANSAC, 5.0)

        h_50, w_50 = reference_50_resized.shape

        pts_50 = np.float32([[0, 0], [0, h_50 - 1], [w_50 - 1, h_50 - 1], [w_50 - 1, 0]]).reshape(-1, 1, 2)
        dst_50 = cv2.perspectiveTransform(pts_50, M_50)

        cv2.polylines(scene_image, [np.int32(dst_50)], True, (0, 255, 0), 3)
        x, y = np.int32(dst_50[0][0])
        cv2.putText(scene_image, '50 Peso Bill', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if not announced_50:
            engine.say("Fifty pesos detected")
            engine.runAndWait()
            announced_50 = True
            announced_100 = False  # Reset flag for the other bill

    # Detect and draw 100 Peso Bill
    if len(good_matches_100) > 10:
        src_pts_100 = np.float32([kp_100[m.queryIdx].pt for m in good_matches_100]).reshape(-1, 1, 2)
        dst_pts_100 = np.float32([kp_scene[m.trainIdx].pt for m in good_matches_100]).reshape(-1, 1, 2)

        M_100, mask_100 = cv2.findHomography(src_pts_100, dst_pts_100, cv2.RANSAC, 5.0)

        h_100, w_100 = reference_100_resized.shape

        pts_100 = np.float32([[0, 0], [0, h_100 - 1], [w_100 - 1, h_100 - 1], [w_100 - 1, 0]]).reshape(-1, 1, 2)
        dst_100 = cv2.perspectiveTransform(pts_100, M_100)

        cv2.polylines(scene_image, [np.int32(dst_100)], True, (0, 255, 0), 3)
        x, y = np.int32(dst_100[0][0])
        cv2.putText(scene_image, '100 Peso Bill', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if not announced_100:
            engine.say("One hundred pesos detected")
            engine.runAndWait()
            announced_100 = True
            announced_50 = False  # Reset flag for the other bill

    # Display live video
    cv2.imshow("Live 50 and 100 Peso Bill Detection", scene_image)

    # Exit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
