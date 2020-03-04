#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
from uwds3_perception.tabletop_perception import TabletopPerception


class TabletopPerceptionNode(object):
    def __init__(self):
        rospy.loginfo("[tabletop_perception] Starting Underworlds perception...")
        self.perception = TabletopPerception()
        rospy.loginfo("[tabletop_perception] Underworlds perception ready !")

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()


if __name__ == '__main__':
    rospy.init_node("uwds3_perception")
    core = TabletopPerceptionNode().run()
