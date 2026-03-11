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
    #define INFO_MSG_MAGENTA(str) do {std::cout << "\033[35m" << str << "\033[0m" << std::endl; } while(false)
    #define INFO_MSG_CYAN(str) do {std::cout << "\033[36m" << str << "\033[0m" << std::endl; } while(false)
    
    #define INFO_MSG_RED_BG(str) do {std::cout << "\033[48;5;1m" << str << "\033[0m" << std::endl; } while(false)
    #define INFO_MSG_GREEN_BG(str) do {std::cout << "\033[48;5;2m" << str << "\033[0m" << std::endl; } while(false)
    #define INFO_MSG_YELLOW_BG(str) do {std::cout << "\033[48;5;3m" << str << "\033[0m" << std::endl; } while(false)
    #define INFO_MSG_BLUE_BG(str) do {std::cout << "\033[48;5;4m" << str << "\033[0m" << std::endl; } while(false)

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
#else
    #define VIS(cmd) do {cmd;} while(false)
    #define BAG(cmd) do {cmd;} while(false)
    #define APPLOG(cmd) do {cmd;} while(false)
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
#else
    #include <ss_timestamp.h>
    #define TimeNow() ss_get_timestamp_us()
    typedef uint64_t TimePoint;

    // return second
    inline double durationSecond(TimePoint t, TimePoint t_before){
        return (double)((int)t - (int)t_before)/1e6;
    }

    // add second to timepoint
    inline TimePoint addDuration(TimePoint t, double& dur){
        return t + dur*1e6;
    }

    static inline int64_t get_thread_time_us(){
        struct timespec tp;
        if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &tp) == 0){
            return (int64_t)(tp.tv_sec * 1e6) + (int64_t)(tp.tv_nsec / 1e3);
        }
        return 0;
    }

#endif



#endif