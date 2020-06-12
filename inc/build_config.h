#include <string>
#pragma once

//https://github.com/gabime/spdlog/wiki/0.-FAQ#how-to-remove-all-debug-statements-at-compile-time-
#undef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"

extern spdlog::logger *logger;
extern spdlog::logger *reporter;
extern std::string globalArgXclBin;
extern std::string globalArgDataPath;
extern unsigned globalBatchsize;
extern bool globalRunTests;
extern bool globalRunClassifier;
extern bool globalDumpTensors;

#define USE_OCL
