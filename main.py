#!/usr/bin/env python3

import rospy
from corrosion_detector import CorrosionDetector

if __name__ == '__main__':
    print("==== Running !!====")
    try:
        rospy.init_node('corrosion_detector_node')
        detector = CorrosionDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in main: {e}")
        import traceback
        traceback.print_exc()