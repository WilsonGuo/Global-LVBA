#include <ros/ros.h>
#include "lvba_system.h"

int main(int argc, char** argv) 
{
    ros::init(argc, argv, "lv_ba");
    ros::NodeHandle nh;
    lvba::LvbaSystem lvba_system(nh);
    lvba_system.runFullPipeline();
    return 0;
}