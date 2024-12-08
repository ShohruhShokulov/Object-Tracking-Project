import cv2
import numpy as np
import time

def main():
    # Prompt user for video file or webcam
    video_path = input("Enter the path to the video file (or press Enter to use the webcam): ").strip()
    if video_path == "":
        cap = cv2.VideoCapture(0)  # Use webcam
    else:
        cap = cv2.VideoCapture(video_path)  # Use provided video file

    # Check if video source is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab first frame.")
        return
    
    # Display list of tracking algorithms
    tracking_algorithms = {
        "1": "CSRT",              # High accuracy, slower
        "2": "KCF",               # Faster, good for objects with minimal occlusion
        "3": "MOSSE",             # Extremely fast, good for simple tracking
        "4": "MEDIANFLOW",        # Robust to abrupt motions
        "5": "Optical Flow"       # Tracks using motion estimation
    }
    
    print("Available tracking algorithms:")
    for key, value in tracking_algorithms.items():
        print(f"{key}: {value}")
    
    # Prompt user to select a tracking algorithm
    algo_choice = input("Select a tracking algorithm by entering the corresponding number: ").strip()
    if algo_choice not in tracking_algorithms:
        print("Invalid choice. Exiting.")
        return
    
    selected_algo = tracking_algorithms[algo_choice]
    print(f"Selected algorithm: {selected_algo}")

    # Initialize the selected tracker
    tracker = None
    bbox = None
    total_frames = 0
    successful_frames = 0

    if selected_algo in ["CSRT", "KCF", "MOSSE", "MEDIANFLOW"]:
        if selected_algo == "CSRT":
            tracker = cv2.legacy.TrackerCSRT_create()
        elif selected_algo == "KCF":
            tracker = cv2.legacy.TrackerKCF_create()
        elif selected_algo == "MOSSE":
            tracker = cv2.legacy.TrackerMOSSE_create()
        elif selected_algo == "MEDIANFLOW":
            tracker = cv2.legacy.TrackerMedianFlow_create()

        # Select Region of Interest (ROI)
        print("Select the object to track and press ENTER or SPACE. Press 'C' to cancel.")
        bbox = cv2.selectROI("Tracking", frame, False)
        cv2.destroyWindow("Tracking")  # Close the ROI selection window
        tracker.init(frame, bbox)
    
    elif selected_algo == "Optical Flow":
        print("Select the object to track and press ENTER or SPACE. Press 'C' to cancel.")
        bbox = cv2.selectROI("Tracking", frame, False)
        cv2.destroyWindow("Tracking")
        old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7))
        mask = np.zeros_like(frame)

    # Start tracking and measure processing time
    start_time = time.time()

    while True:
        # Read a new frame
        ret, frame = cap.read()
        if not ret:
            print("End of video or camera feed.")
            break
        
        total_frames += 1

        if selected_algo in ["CSRT", "KCF", "MOSSE", "MEDIANFLOW"]:
            success, bbox = tracker.update(frame)
            if success:
                successful_frames += 1
                x, y, w, h = [int(i) for i in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Tracking: {selected_algo}", (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Lost", (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        elif selected_algo == "Optical Flow":
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)))
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            for new, old in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
                cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
            frame = cv2.add(frame, mask)
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        # Display the result
        cv2.imshow("Tracking", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Measure end time and calculate FPS
    end_time = time.time()
    total_time = end_time - start_time
    fps = total_frames / total_time if total_time > 0 else 0

    # Calculate accuracy
    accuracy = (successful_frames / total_frames) * 100 if total_frames > 0 else 0

    # Display results
    print(f"Results for {selected_algo} Tracker:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Processing Time (FPS): {fps:.2f} frames/second")

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
