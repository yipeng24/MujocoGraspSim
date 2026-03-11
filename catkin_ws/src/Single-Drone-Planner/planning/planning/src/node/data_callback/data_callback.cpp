#include <node/data_callback_node.h>

using rot_util = rotation_util::RotUtil;

DataCallBacks::DataCallBacks(std::shared_ptr<ShareDataManager> dataManagerPtr,
                             std::shared_ptr<parameter_server::ParaeterSerer> paraPtr):
                             dataManagerPtr_(dataManagerPtr),
                             paraPtr_(paraPtr){

}

