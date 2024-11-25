# 
# 设置CMake的最低版本要求为3.9
cmake_minimum_required(VERSION 3.9)  
# 项目名称为demo
project(demo)  
 
 
# 设定编译参数
# set(CMAKE_CXX_STANDARD 11)
# CMAKE_BUILD_TYPE 设置为 "Debug"，启用调试模式（生成带有调试信息的可执行文件）
set(CMAKE_BUILD_TYPE "Debug")
 
# 设定源码列表.cpp。SOURCE_FILES 和 Test_file 分别包含单个源文件路径
# 定义一个变量 SOURCE_FILES，并将其值设置为相对路径 ./excute/nn.cpp
# 这个变量包含了一个源文件路径，该源文件将在后续的 CMake 配置中被引用，用于生成可执行文件或库
set(SOURCE_FILES ./excute/nn.cpp)
# 定义一个变量 Test_file，并将其值设置为相对路径 ./excute/main.cpp
# 这个变量包含了一个源文件路径，该源文件也将在后续的 CMake 配置中被引用，用于生成可执行文件或进行其他配置
# set(Test_file ./excute/main.cpp)

# 设定所有源码列表 ：aux_source_directory(<dir> <variable>)
# 比如:将${CMAKE_SOURCE_DIR}目录下，也就是最顶级目录下所有的.cpp文件放入DIR变量中，后面的add_executable就可以很简化
# 使用 aux_source_directory 将 ${CMAKE_SOURCE_DIR}/src 目录下的所有 .cpp 文件放入 DIR 变量中
aux_source_directory(${CMAKE_SOURCE_DIR}/src DIR)  

# 将 DIR 中的源文件编译成可执行文件 hello_world
# add_executable(hello_world ${DIR})
 
 
# 设定头文件路径。include_directories 用于添加多个头文件搜索路径
include_directories(tfhe/include/)
include_directories(include/)
include_directories(/home/glf/usr/local/include/tfhe)
# include_directories("路径1"  “路径2”...)
 
 
# 设定链接库的路径（一般使用第三方非系统目录下的库）
# link_directories 用于添加多个链接库路径
# link_directories(/home/glf/usr/local/lib/libtfhe-nayuki-avx.so)
# link_directories(/home/glf/usr/local/lib/libtfhe-nayuki-portable.so)
# link_directories(/home/glf/usr/local/lib/libtfhe-spqlios-avx.so)
# link_directories(/home/glf/usr/local/lib/libtfhe-spqlios-fma.so)
   
# link_directories("路径1"  “路径2”...)
 
 
# 添加子目录,作用相当于进入子目录里面，展开子目录的CMakeLists.txt
# 同时执行，子目录中的CMakeLists.txt一般是编译成一个库，作为一个模块
# 在父目录中可以直接引用子目录生成的库
# add_subdirectory(math)
 
 
# 生成动/静态库
# add_library 用于定义库的名称、类型（动态或静态）及源文件列表
# add_library(动/静态链接库名称  SHARED/STATIC(可选，默认STATIC)  源码列表)
# 可以单独生成多个模块
 
 
# 生成可执行文件
# add_executable 用于定义可执行文件的名称及其源文件列表。这里生成了 enc_snn 和 enc_test 两个可执行文件
add_executable(enc_snn  ${SOURCE_FILES}  ${DIR})
# add_executable(enc_test  ${Test_file} ${DIR})
#!比如：add_executable(hello_world    ${SOURCE_FILES})
 

# 就是g++ 编译选项中-l后的内容，不要有多余空格
# 为可执行文件 enc_snn 指定链接库。target_link_libraries 用于添加库文件路径，这些路径在链接过程中会被使用
target_link_libraries(enc_snn
${CMAKE_SOURCE_DIR}/tfhe/lib/libtfhe-nayuki-avx.so 
${CMAKE_SOURCE_DIR}/tfhe/lib/libtfhe-nayuki-portable.so 
${CMAKE_SOURCE_DIR}/tfhe/lib/libtfhe-spqlios-avx.so 
${CMAKE_SOURCE_DIR}/tfhe/lib/libtfhe-spqlios-fma.so
pthread)

# 为可执行文件 enc_test 指定链接库。target_link_libraries 用于添加库文件路径，这些路径在链接过程中会被使用
# target_link_libraries(enc_test
# ${CMAKE_SOURCE_DIR}/tfhe/lib/libtfhe-nayuki-avx.so 
# ${CMAKE_SOURCE_DIR}/tfhe/lib/libtfhe-nayuki-portable.so 
# ${CMAKE_SOURCE_DIR}/tfhe/lib/libtfhe-spqlios-avx.so 
# ${CMAKE_SOURCE_DIR}/tfhe/lib/libtfhe-spqlios-fma.so
# pthread
# )


# 用于执行自定义Shell命令。ADD_CUSTOM_COMMAND 定义了一个在指定目标文件构建之后执行的命令。这里的例子是将生成的 myLevelDB 文件复制到上一级目录
#! 执行shell命令
# ADD_CUSTOM_COMMAND(
#           TARGET myLevelDB 
#           POST_BUILD #在目标文件myLevelDBbuild之后，执行下面的拷贝命令，还可以选择PRE_BUILD命令将会在其他依赖项执行前执行  PRE_LINK命令将会在其他依赖项执行完后执行  POST_BUILD命令将会在目标构建完后执行。
#           COMMAND cp ./myLevelDB  ../
# ) 