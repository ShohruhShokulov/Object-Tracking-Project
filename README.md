# Object-Tracking-Project
Object Tracking Project from Digital Image Processing course
# **Object Tracking Using OpenCV**

### **Author:** Shohruh Shokulov  
**Student ID:** 12225260  

---

## **Project Overview**
This project demonstrates object tracking using OpenCV's **Channel and Spatial Reliability Tracking (CSRT)**. The program allows users to track objects in pre-recorded video files or live video streams (webcam). The CSRT algorithm ensures accurate and robust tracking, even in challenging scenarios such as partial occlusion, dynamic backgrounds, or changes in object appearance.

---

## **Features**
- **Real-time Object Tracking**: Tracks objects live using a webcam.  
- **Video File Tracking**: Tracks objects in a provided video file.  
- **Interactive ROI Selection**: Users can select the object manually in the first frame.  
- **Visual Feedback**:  
  - **Green bounding box**: Successful tracking.  
  - **Red message ("Lost")**: When tracking fails.  

---

## **Algorithm Explanation**
The **CSRT (Channel and Spatial Reliability Tracking)** algorithm is a robust object tracking method provided by OpenCV. It analyzes spatial and channel reliability to adapt to changes in an object’s size, shape, or appearance, making it suitable for:
- Tracking objects undergoing complex movements.
- Handling dynamic or cluttered backgrounds.
- Adapting to partial occlusion or deformation.

Compared to faster trackers like KCF, CSRT provides higher accuracy at the cost of slightly slower performance. It is ideal for applications requiring precision.

---

## **Usage Instructions**


### **1. Install Dependencies**
Install the required Python libraries using:
```bash
pip install -r requirements.txt
```

### **2. Run the Script**
Run the script to start object tracking:
```bash
python object_tracker.py
```

### **3. Provide Input**
- To use a **video file**, enter its path when prompted (e.g., `data/sample_video.mp4`).  
- To use the **webcam**, press Enter.

### **4. Track the Object**
- Use the mouse to select the object in the first frame.  
- Press **Enter** or **Space** to confirm the selection.  
- The tracker will follow the object frame by frame.  
- Press **'q'** to stop tracking.
```
