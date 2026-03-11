#ifndef UTIL_GYM_H
#define UTIL_GYM_H

#include <chrono>
#include <ios>
#include <cstdlib>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

#define ROS
// #define SS_DBUS

#define INFO
// #define DEBUG

#ifdef DEBUG
    #define DEBUG_MSG(str) do {std::cout << str << std::endl; } while(false)
    #ifdef ROS
        #define DEBUG_VIS(cmd) do {cmd;} while(false)
    #else
         #define DEBUG_VIS(cmd) do {} while(false)
         #endif
#else
    #define DEBUG_MSG(cmd) do {} while(false)
    #define DEBUG_VIS(cmd) do {} while(false)
#endif

#ifdef INFO
    #define INFO_MSG(str) do {std::cout << str << std::endl; } while(false)
    #define INFO_MSG_RED(str) do {std::cout << "\033[31m" << str << "\033[0m" << std::endl; } while(false)
    #define INFO_MSG_GREEN(str) do {std::cout << "\033[32m" << str << "\033[0m" << std::endl; } while(false)
    #define INFO_MSG_YELLOW(str) do {std::cout << "\033[33m" << str << "\033[0m" << std::endl; } while(false)
    #define INFO_MSG_BLUE(str) do {std::cout << "\033[34m" << str << "\033[0m" << std::endl; } while(false)
#else
    #define INFO_MSG(str) do {} while(false)
    #define INFO_MSG_RED(str) do {} while(false)
    #define INFO_MSG_GREEN(str) do {} while(false)
    #define INFO_MSG_YELLOW(str) do {} while(false)
    #define INFO_MSG_BLUE(str) do {} while(false)
#endif

#ifdef ROS
    #define VIS(cmd) do {cmd;} while(false)
    #define BAG(cmd) do {} while(false)
    #define APPLOG(cmd) do {} while(false)
#endif

#ifdef ROS
    #define TimeNow() std::chrono::high_resolution_clock::now()
    typedef std::chrono::high_resolution_clock::time_point TimePoint;

    // return second
    inline double durationSecond(TimePoint t, TimePoint t_before){
        return std::chrono::duration_cast<std::chrono::duration<double>>(t - t_before).count();
    }

    // add second to timepoint
    inline TimePoint addDuration(TimePoint t, double& dur){
        return t + std::chrono::milliseconds((int)(dur*1e3));
    }
#endif



#endif