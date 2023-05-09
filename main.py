import cv2
import mediapipe as mp

# initialize the camera
camera = cv2.VideoCapture(0)  # 0 is the default camera index
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# initialize the Mediapipe drawing utility
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# define the size and position of the square
square_size = 50
square_x = 295
square_y = 215

# loop over the frames from the camera
while True:
    # read a frame from the camera
    ret, image = camera.read()

    # flip the image horizontally for a mirror effect
    image = cv2.flip(image, 1)

    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect hands in the image
    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7) as hands:
        results = hands.process(gray)

        # get the landmarks of the hand
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # draw the landmarks on the image
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # get the top points of the index, thumb, and middle fingers
            index_top = tuple(
                map(int, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image.shape[1],
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0]))
            thumb_top = tuple(
                map(int, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image.shape[1],
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image.shape[0]))
            middle_top = tuple(
                map(int, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image.shape[1],
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image.shape[0]))

            # check if any of the top points is inside the square
            if index_top[0] >= square_x and index_top[0] <= square_x + square_size and index_top[1] >= square_y and index_top[1] <= square_y + square_size:
                square_x = index_top[0] - square_size // 2
                square_y = index_top[1] - square_size // 2
            elif thumb_top[0] >= square_x and thumb_top[0] <= square_x + square_size and thumb_top[1] >= square_y and thumb_top[1] <= square_y + square_size:
                square_x = thumb_top[0] - square_size // 2
                square_y = thumb_top[1] - square_size // 2
            elif middle_top[0] >= square_x and middle_top[0] <= square_x + square_size and middle_top[1] >= square_y and middle_top[1] <= square_y + square_size:
                square_x = middle_top[0] - square_size // 2
                square_y = middle_top[1] - square_size // 2
 # draw the square on the image
                cv2.rectangle(
                    image, (square_x, square_y),
                    (square_x + square_size, square_y + square_size),
                    (0, 255, 0), 2)

    # display the image
    cv2.imshow("Hand Tracking Game", image)

    # check if the user wants to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the camera and close the window
camera.release()
cv2.destroyAllWindows()
