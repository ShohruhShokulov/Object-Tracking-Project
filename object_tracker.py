import cv2

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
    
    # Select Region of Interest (ROI)
    print("Select the object to track and press ENTER or SPACE. Press 'C' to cancel.")
    bbox = cv2.selectROI("Tracking", frame, False)
    cv2.destroyWindow("Tracking")  # Close the ROI selection window

    # Initialize tracker
    tracker = cv2.TrackerCSRT.create()  # High-accuracy tracker
    tracker.init(frame, bbox)
    
    while True:
        # Read a new frame
        ret, frame = cap.read()
        if not ret:
            print("End of video or camera feed.")
            break
        
        # Update tracker
        success, bbox = tracker.update(frame)
        
        # Draw bounding box
        if success:
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Lost", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the result
        cv2.imshow("Tracking", frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
