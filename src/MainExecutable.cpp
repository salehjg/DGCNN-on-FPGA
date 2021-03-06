#include "build_config.h"
#include "argparse.h"
#include "ClassifierMultiPlatform.h"
#include "ocl_imp/xilinx/XilinxImpUnitTests.h"
#include <iostream>
#include <execinfo.h>
#include <unistd.h>
#include <string>
#include <csignal>

using namespace std;
using namespace argparse;

spdlog::logger *logger;
spdlog::logger *reporter;
string globalArgXclBin;
string globalArgDataPath;
unsigned globalBatchsize;
bool globalRunTests=false;
bool globalRunClassifier=true;
bool globalDumpTensors=false;

void handler(int sig) {
    void *array[40];
    size_t size;

    // get void*'s for all entries on the stack
    size = backtrace(array, 40);

    // print out all the frames to stderr
    cerr<<"The host program has crashed, printing call stack:\n";
    cerr<<"Error: signal "<< sig<<"\n";
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    
    SPDLOG_LOGGER_CRITICAL(logger,"The host program has crashed.");
    spdlog::shutdown();
    
    exit(SIGSEGV);
}

void handlerInt(int sig_no)
{
    SPDLOG_LOGGER_CRITICAL(logger,"CTRL+C pressed, terminating...");
    spdlog::shutdown();
    exit(SIGINT);
}

int main(int argc, const char* argv[]){
    signal(SIGSEGV, handler);
    signal(SIGABRT, handler);
    signal(SIGINT, handlerInt);

    ArgumentParser parser("ArgParse");
    parser.add_argument()
        .names({"-i", "--image"})
        .description("FPGA image(*.xclbin or *.awsxclbin)")
        .required(true);

    parser.add_argument()
        .names({"-d", "--data"})
        .description("Data directory")
        .required(true);  

    parser.add_argument()
        .names({"-b", "--batchsize"})
        .description("Batch-size")
        .required(false);     
          
    parser.add_argument()
        .names({"-e", "--emumode"})
        .description("Forced emulation mode(sw_emu or hw_emu)")
        .required(false);  

    parser.add_argument()
        .names({"-t", "--testsonly"})
        .description("Only run OCl tests(no value is needed for this argument)")
        .required(false);

    parser.add_argument()
        .names({"-c", "--classifieronly"})
        .description("Only run OCl classifier(no value is needed for this argument)")
        .required(false);   

    parser.add_argument()
        .names({"-k", "--dumptensors"})
        .description("Dump tensors into *.npy files in the data directory(no value is needed for this argument)")
        .required(false);     

    parser.add_argument()
        .names({"-n", "--nolog"})
        .description("Disable logging.(no value is needed for this argument)")
        .required(false);      

    parser.enable_help();
    auto err = parser.parse(argc, argv);
    if(err){
        std::cerr << err << std::endl;
        parser.print_help();
        return -1;
    }

    if(parser.exists("help")){
        parser.print_help();
        return 0;
    }

    {
        // HOST LOGGER
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::trace);
        console_sink->set_pattern("[%H:%M:%S.%e][%^%l%$] %v");

        auto file_sink1 = std::make_shared<spdlog::sinks::basic_file_sink_mt>("hostlog_0trace.log", true);
        file_sink1->set_level(spdlog::level::trace);
        file_sink1->set_pattern("[%H:%M:%S.%f][%^%l%$][source %s][function %!][line %#] %v");

        auto file_sink0 = std::make_shared<spdlog::sinks::basic_file_sink_mt>("hostlog_1debug.log", true);
        file_sink0->set_level(spdlog::level::debug);
        file_sink0->set_pattern("[%H:%M:%S.%f][%^%l%$][source %s][function %!][line %#] %v");

        auto file_sink2 = std::make_shared<spdlog::sinks::basic_file_sink_mt>("hostlog_2info.log", true);
        file_sink2->set_level(spdlog::level::info);
        file_sink2->set_pattern("[%H:%M:%S.%f][%^%l%$][source %s][function %!][line %#] %v");

        auto file_sink3 = std::make_shared<spdlog::sinks::basic_file_sink_mt>("hostlog_3wraning.log", true);
        file_sink3->set_level(spdlog::level::warn);
        file_sink3->set_pattern("[%H:%M:%S.%f][%^%l%$][source %s][function %!][line %#] %v");

        logger = new spdlog::logger("DP1FPGA Host-logger", {console_sink, file_sink0, file_sink1, file_sink2, file_sink3});
        logger->set_level(spdlog::level::trace); 


        // PERFORMANCE REPORTER
        auto file_sink4 = std::make_shared<spdlog::sinks::basic_file_sink_mt>("report_kernel.log", true); //Kernels are reported
        file_sink4->set_level(spdlog::level::info);
        file_sink4->set_pattern("[%H:%M:%S.%f][%^%l%$][source %s][function %!][line %#] %v");

        auto file_sink5 = std::make_shared<spdlog::sinks::basic_file_sink_mt>("report_host_kernel.log", true); //kernels and host
        file_sink5->set_level(spdlog::level::debug);
        file_sink5->set_pattern("[%H:%M:%S.%f][%^%l%$][source %s][function %!][line %#] %v");

        // this is strictly used to log host and device performance.
        reporter = new spdlog::logger("DP1FPGA Host-reporter", {file_sink4,file_sink5});
        reporter->set_level(spdlog::level::trace); 


        if(parser.exists("n")) {
            logger->set_level(spdlog::level::off); 
            reporter->set_level(spdlog::level::off); 
        }
        //SPDLOG_LOGGER_TRACE(logger,"test log ::: trace");
        //SPDLOG_LOGGER_DEBUG(logger,"test log ::: debug");
        //SPDLOG_LOGGER_INFO(logger,"test log ::: info");
        //SPDLOG_LOGGER_WARN(logger,"test log ::: warn");
        //SPDLOG_LOGGER_ERROR(logger,"test log ::: error");
        //SPDLOG_LOGGER_CRITICAL(logger,"test log ::: critical");

        //SPDLOG_LOGGER_INFO(reporter,"test log ::: info");
        //SPDLOG_LOGGER_DEBUG(reporter,"test log ::: debug");
        
    }

    if(parser.exists("b")) {
        globalBatchsize = parser.get<unsigned>("b");
    }else{
        globalBatchsize = 5;
    }
    SPDLOG_LOGGER_INFO(logger,"Batch-size: {}", globalBatchsize);

    if(parser.exists("i")) {
        globalArgXclBin = parser.get<string>("i").c_str();
        SPDLOG_LOGGER_INFO(logger,"FPGA Image: {}", globalArgXclBin);
    }

    if(parser.exists("d")) {
        globalArgDataPath = parser.get<string>("d");
        SPDLOG_LOGGER_INFO(logger,"Data Directory: {}", globalArgDataPath);
    }

    if(parser.exists("e")) {
        const char *forcedMode = parser.get<string>("e").c_str();
        SPDLOG_LOGGER_INFO(logger,"Forced Emulation Mode: {}", forcedMode);
        if (setenv("XCL_EMULATION_MODE", forcedMode, 1) < 0) {
            std::cerr <<""<<std::endl;
            SPDLOG_LOGGER_ERROR(logger,"Can not set env var XCL_MODE.");
        }
    }

    if(parser.exists("testsonly")) {
        globalRunClassifier = false;
        globalRunTests = true; 
        SPDLOG_LOGGER_INFO(logger,"Only OCl tests are going to be run.");
    }

    if(parser.exists("classifieronly")) {
        globalRunClassifier = true;
        globalRunTests = false;
        SPDLOG_LOGGER_INFO(logger,"Only OCl classifier is going to be run.");
    }

    if(parser.exists("dumptensors")) {
        globalDumpTensors = true;
        SPDLOG_LOGGER_INFO(logger,"Tensors will be dumped into separate numpy files in the data directory.");
    }

    if(globalRunTests){
        SPDLOG_LOGGER_INFO(logger,"======================================================");
        SPDLOG_LOGGER_INFO(logger,"Running Kernel Unit Tests ...");
        XilinxImpUnitTests xilinxImpUnitTests;
        xilinxImpUnitTests.RunAll();
    }

    if(globalRunClassifier){
        SPDLOG_LOGGER_INFO(logger,"======================================================");
        SPDLOG_LOGGER_INFO(logger,"Running Selected ModelArch ...");
        ClassifierMultiplatform();
    }





}
