import cv2
import mediapipe as mp

# Function to classify Tree Pose
def classify_tree_pose(landmarks):
    left_hand = landmarks[mp.solutions.holistic.HandLandmark.LEFT_INDEX]
    right_hand = landmarks[mp.solutions.holistic.HandLandmark.RIGHT_INDEX]
    left_foot = landmarks[mp.solutions.holistic.PoseLandmark.LEFT_ANKLE]
    right_foot = landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_ANKLE]

    if all(left_hand) and all(right_hand) and not all(left_foot) and not all(right_foot):
        return True
    else:
        return False

# Function to classify Warrior Pose
def classify_warrior_pose(landmarks):
    left_hand = landmarks[mp.solutions.holistic.HandLandmark.LEFT_INDEX]
    right_hand = landmarks[mp.solutions.holistic.HandLandmark.RIGHT_INDEX]
    left_foot = landmarks[mp.solutions.holistic.PoseLandmark.LEFT_ANKLE]
    right_foot = landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_ANKLE]

    if all(left_hand) and all(right_hand) and all(left_foot) and all(right_foot):
        return True
    else:
        return False

# Function to classify Downward Dog Pose
def classify_downward_dog_pose(landmarks):
    left_hand = landmarks[mp.solutions.holistic.HandLandmark.LEFT_INDEX]
    right_hand = landmarks[mp.solutions.holistic.HandLandmark.RIGHT_INDEX]
    left_foot = landmarks[mp.solutions.holistic.PoseLandmark.LEFT_ANKLE]
    right_foot = landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_ANKLE]

    if not all(left_hand) and not all(right_hand) and all(left_foot) and all(right_foot):
        return True
    else:
        return False

def classify_uttanasana_pose(landmarks):
    # Landmark indices for hips, knees, and ankles
    left_hip = landmarks[mp.solutions.holistic.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_HIP]
    left_knee = landmarks[mp.solutions.holistic.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_KNEE]
    left_ankle = landmarks[mp.solutions.holistic.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_ANKLE]

    if(left_hip and right_hip and left_knee and right_knee and left_knee and left_ankle and right_ankle):
        return True
    else:
        return False

# Function to classify Cobra Pose
def classify_cobra_pose(landmarks):
    left_elbow = landmarks[mp.solutions.holistic.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_ELBOW]

    if all(left_elbow) and all(right_elbow):
        return True
    else:
        return False

# Function to classify Child's Pose
def classify_childs_pose(landmarks):
    left_elbow = landmarks[mp.solutions.holistic.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks[mp.solutions.holistic.PoseLandmark.RIGHT_ELBOW]

    if left_elbow and right_elbow:
        return True
    else:
        return False

    

# Function to detect yoga poses
def detect_yoga_poses(frame):
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
      
##            if classify_tree_pose(landmarks):
##                cv2.putText(frame, "Tree Pose", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
##            elif classify_warrior_pose(landmarks):
##                cv2.putText(frame, "Warrior Pose", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
##            elif classify_downward_dog_pose(landmarks):
##                cv2.putText(frame, "Downward Dog Pose", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if classify_uttanasana_pose(landmarks):
                cv2.putText(frame, "Tree Pose", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif classify_cobra_pose(landmarks):
                cv2.putText(frame, "Cobra Pose", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif classify_childs_pose(landmarks):
                cv2.putText(frame, "Child's Pose", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame

def main():
    cap = cv2.VideoCapture(0)  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = detect_yoga_poses(frame)

        cv2.imshow('Yoga Poses Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
